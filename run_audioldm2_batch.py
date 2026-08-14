import os
import sys
from typing import Literal
import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from diffusers import AudioLDM2Pipeline

# --- MONKEY PATCH FOR TRANSFORMERS / AUDIOLDM2 ---
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

def _patch_update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
    model_kwargs["past_key_values"] = getattr(outputs, "past_key_values", None)
    if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
        mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [mask, mask.new_ones((mask.shape[0], 1))], dim=-1
        )
    return model_kwargs

GPT2Model._update_model_kwargs_for_generation = _patch_update_model_kwargs_for_generation


# --- MODULE 1: THE SCORE-BASED MIXER (16 kHz Native) ---
def intelligent_weighted_mix(
    audio_data: dict, 
    folder_path: str, 
    num_audio_mix: int = 3, 
    sr: int = 16000, 
    weight_by: Literal["softmax_score", "cosine_sim"] = "softmax_score"
):
    target_samples = 10 * sr  # 160,000 samples for AudioLDM 2 (10s @ 16kHz)
    final_mix = torch.zeros((1, target_samples))

    for filename, scores in list(audio_data.items())[:num_audio_mix]:
        actual_wav_name = str(filename).replace('.npy', '.wav')
        path = os.path.join(folder_path, actual_wav_name)
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: {actual_wav_name} not found. Skipping.")
            continue

        data, orig_sr = sf.read(path)
        wf = torch.from_numpy(data).float()
        
        if wf.ndim == 1:
            wf = wf.unsqueeze(0)
        else:
            wf = wf.T
        
        if wf.shape[0] > 1:
            wf = torch.mean(wf, dim=0, keepdim=True)

        if orig_sr != sr:
            wf = T.Resample(orig_sr, sr)(wf)
        
        wf = wf[:, :target_samples]
        if wf.shape[1] < target_samples:
            wf = torch.nn.functional.pad(wf, (0, target_samples - wf.shape[1]))
            
        energy = torch.sqrt(torch.mean(wf**2)) + 1e-8
        score = scores[weight_by] if isinstance(scores, dict) else float(scores)
        balanced_wf = (wf / energy) * score
        
        final_mix += balanced_wf

    max_val = torch.max(torch.abs(final_mix))
    if max_val > 0:
        final_mix = final_mix / (max_val + 1e-8)

    return final_mix, sr


# --- MODULE 2: CORRECT 16kHz VAE MEL-SPECTROGRAM CONVERSION ---
def wav_to_vae_latents(waveform: torch.Tensor, vae, device: str = "cuda"):
    """
    Computes the exact log-mel spectrogram (16kHz, 64 mel bins, hop 160)
    expected by AudioLDM / AudioLDM 2 VAE.
    """
    # Ensure waveform is on GPU and shape is [batch, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        waveform = waveform.squeeze(1)

    waveform = waveform.to(device)

    # AudioLDM2 VAE Mel-Spectrogram specs
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        f_min=0,
        f_max=8000,
        n_mels=64,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
    ).to(device)

    # 1. Extract Mel Spectrogram -> shape [batch, 64, time_steps]
    mel_spec = mel_transform(waveform)

    # 2. Log-compression (standard for AudioLDM VAE)
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    # 3. Shape to [batch, 1, time_steps, 64] for VAE encoder
    log_mel_spec = log_mel_spec.transpose(1, 2).unsqueeze(1).to(dtype=vae.dtype)

    # 4. VAE Encode
    init_latents = vae.encode(log_mel_spec).latent_dist.sample()
    init_latents = init_latents * vae.config.scaling_factor
    return init_latents


# --- MODULE 3: AUDIO-TO-AUDIO DIFFUSION ---
def run_audioldm2_audio2audio(
    pipe: AudioLDM2Pipeline,
    mixed_waveform: torch.Tensor,
    text_prompt: str,
    negative_prompt: str = "low quality, distorted, noisy, glitch",
    strength: float = 0.45,
    num_inference_steps: int = 100,
    guidance_scale: float = 3.5,
    device: str = "cuda"
):
    with torch.no_grad():
        # 1. Encode Text Prompts
        prompt_embeds, prompt_mask, gen_prompt_embeds = pipe.encode_prompt(
            prompt=text_prompt,
            device=device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=(guidance_scale > 1.0),
            negative_prompt=negative_prompt,
        )

        # 2. Correctly Encode 16kHz Audio into VAE Latents
        init_latents = wav_to_vae_latents(mixed_waveform, pipe.vae, device=device)

        # 3. Set Scheduler Timesteps & Noise
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start_idx = max(num_inference_steps - init_timestep, 0)
        timesteps_to_use = timesteps[t_start_idx:]
        start_timestep = timesteps_to_use[0:1]

        # 4. Add Noise matching start_timestep
        noise = torch.randn_like(init_latents)
        latents = pipe.scheduler.add_noise(init_latents, noise, start_timestep)

        # 5. Denoise Loop
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)

        for t in timesteps_to_use:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=gen_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=prompt_mask,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # 6. Decode Latents -> Mel Spectrogram -> 16kHz Audio
        latents = latents / pipe.vae.config.scaling_factor
        mel_spectrogram = pipe.vae.decode(latents).sample
        
        while mel_spectrogram.dim() > 3:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        audio = pipe.vocoder(mel_spectrogram)
        audio = audio.squeeze().cpu().float().numpy()

    return audio


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    search_results = {
        '-jxXv_SoNaI.npy': {
            '13wjsR-VT_o.npy': {'softmax_score': 0.42002373933792114, 'cosine_sim': 0.18692827224731445},
            '0q53YxUMkKw.npy': {'softmax_score': 0.24088945984840393, 'cosine_sim': 0.18136852979660034},
            '-BAKe6QGTUk.npy': {'softmax_score': 0.13764320313930511, 'cosine_sim': 0.1757718026638031},
        }
    }

    input_folder = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
    output_folder = os.environ.get("OUTPUT_DIR", "./test_outputs")
    os.makedirs(output_folder, exist_ok=True)

    query_key = list(search_results.keys())[0]
    candidates_dict = search_results[query_key]
    video_id = os.path.splitext(query_key)[0]

    # Native sampling rate for AudioLDM 2
    NATIVE_SR = 16000

    print("📊 Mixing candidate audio files at 16kHz...")
    mixed_wf, sr = intelligent_weighted_mix(
        audio_data=candidates_dict,
        folder_path=input_folder,
        num_audio_mix=3,
        sr=NATIVE_SR,
        weight_by="softmax_score"
    )

    # Save RAW Mix (16kHz PCM)
    raw_path = os.path.join(output_folder, f"{video_id}_RAW_MIX.wav")
    sf.write(raw_path, mixed_wf.squeeze().cpu().numpy(), sr, subtype="PCM_16")
    print(f"📁 Raw mix saved: {raw_path}")

    # Load AudioLDM2 Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Loading AudioLDM 2 pipeline on {device}...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Run Generation
    prompt = "A high quality acoustic instrument melody matching the scene"
    print(f"✨ Running True Audio-to-Audio Diffusion at 16kHz...")
    
    generated_audio = run_audioldm2_audio2audio(
        pipe=pipe,
        mixed_waveform=mixed_wf,
        text_prompt=prompt,
        strength=0.45,   # Adjust between 0.35 (close to mix) and 0.6 (more creative)
        num_inference_steps=100,
        guidance_scale=3.5,
        device=device
    )

    # Save Final Result (16kHz PCM)
    out_path = os.path.join(output_folder, f"{video_id}_GEN.wav")
    generated_audio = np.clip(generated_audio, -1.0, 1.0)
    sf.write(out_path, generated_audio, sr, subtype="PCM_16")
    print(f"✅ Success! Crystal-clear audio saved: {out_path}")