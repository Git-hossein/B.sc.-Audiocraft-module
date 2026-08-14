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


# --- MODULE 1: THE SCORE-BASED MIXER ---
def intelligent_weighted_mix(
    audio_data: dict, 
    folder_path: str, 
    num_audio_mix: int = 3, 
    sr: int = 48000, 
    weight_by: Literal["softmax_score", "cosine_sim"] = "softmax_score"
):
    target_samples = 10 * sr
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


# --- MODULE 2: TRUE AUDIO-TO-AUDIO DIFFUSION ---
def run_audioldm2_audio2audio(
    pipe: AudioLDM2Pipeline,
    mixed_waveform: torch.Tensor,
    text_prompt: str,
    negative_prompt: str = "low quality, distorted, noisy, glitch",
    strength: float = 0.5,
    num_inference_steps: int = 200,
    guidance_scale: float = 3.5,
    sr: int = 48000,
    device: str = "cuda"
):
    """
    Proper Audio-to-Audio Diffusion:
    1. Extracts mel-spectrogram & encodes into VAE latent space
    2. Sets scheduler timesteps and finds the exact start index corresponding to strength
    3. Adds noise ONLY for the starting timestep
    4. Runs denoising ONLY over the remaining timesteps (t_start -> 0)
    5. Decodes latents back to audio via Vocoder
    """
    waveform = mixed_waveform.to(device=device, dtype=pipe.vae.dtype)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        # 1. Encode prompt & negative prompt
        prompt_embeds, prompt_mask, gen_prompt_embeds = pipe.encode_prompt(
            prompt=text_prompt,
            device=device,
            num_waveforms_per_prompt=1,
            do_classifier_free_guidance=(guidance_scale > 1.0),
            negative_prompt=negative_prompt,
        )

        # 2. Extract Mel-Spectrogram and encode to VAE Latents
        inputs = pipe.feature_extractor(
            raw_speech=waveform.squeeze().cpu().numpy(), 
            sampling_rate=sr, 
            return_tensors="pt"
        )
        mel = inputs.input_features.to(device=device, dtype=pipe.vae.dtype)
        init_latents = pipe.vae.encode(mel).latent_dist.sample()
        init_latents = init_latents * pipe.vae.config.scaling_factor

        # 3. Calculate timesteps for partial denoising
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # Determine start timestep based on strength
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start_idx = max(num_inference_steps - init_timestep, 0)
        timesteps_to_use = timesteps[t_start_idx:]
        start_timestep = timesteps_to_use[0:1]

        # 4. Inject noise matching the starting timestep
        noise = torch.randn_like(init_latents)
        latents = pipe.scheduler.add_noise(init_latents, noise, start_timestep)

        # 5. Denoise loop (ONLY from t_start to 0)
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)

        for t in timesteps_to_use:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=gen_prompt_embeds,
                encoder_hidden_states_1=prompt_embeds,
                encoder_attention_mask_1=prompt_mask,
                return_dict=False,
            )[0]

            # Guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Step scheduler
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # 6. Decode latents to Mel-spectrogram & Vocode to Waveform
        latents = latents / pipe.vae.config.scaling_factor
        mel_spectrogram = pipe.vae.decode(latents).sample
        audio = pipe.vocoder(mel_spectrogram)

        # Post-process to 1D float array
        audio = audio.squeeze().cpu().float().numpy()

    return audio


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    search_results = {
        '-jxXv_SoNaI.npy': 
        {'13wjsR-VT_o.npy': {'softmax_score': 0.42002373933792114,
                                         'cosine_sim': 0.18692827224731445},
                     '0q53YxUMkKw.npy': {'softmax_score': 0.24088945984840393,
                                         'cosine_sim': 0.18136852979660034},
                     '-BAKe6QGTUk.npy': {'softmax_score': 0.13764320313930511,
                                         'cosine_sim': 0.1757718026638031},
                     '2Sr__ctC3s4.npy': {'softmax_score': 0.11249998956918716,
                                         'cosine_sim': 0.17375469207763672},
                     '1nZWM7d70Vk.npy': {'softmax_score': 0.08894357085227966,
                                         'cosine_sim': 0.17140518128871918}
                                         }
                        }

    input_folder = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
    output_folder = os.environ.get("OUTPUT_DIR", "./test_outputs")
    os.makedirs(output_folder, exist_ok=True)

    query_key = list(search_results.keys())[0]
    candidates_dict = search_results[query_key]
    video_id = os.path.splitext(query_key)[0]

    print(f"🎬 Processing Query: {query_key}")

    # 1. Mix input files
    print("📊 Mixing candidate files at 48kHz...")
    mixed_wf, sr = intelligent_weighted_mix(
        audio_data=candidates_dict,
        folder_path=input_folder,
        num_audio_mix=3,
        sr=48000,
        weight_by="softmax_score"
    )

    # Save RAW mix as standard 16-bit PCM
    raw_path = os.path.join(output_folder, f"{video_id}_RAW_MIX.wav")
    sf.write(raw_path, mixed_wf.squeeze().cpu().numpy(), sr, subtype="PCM_16")
    print(f"📁 Raw mix saved: {raw_path}")

    # 2. Load Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Loading AudioLDM 2 pipeline onto {device}...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # 3. Generate Audio with true Audio-to-Audio partial diffusion
    prompt = "A high quality acoustic instrument melody matching the scene"
    print(f"✨ Running Audio-to-Audio Diffusion (Strength = 0.5)...")
    
    generated_audio = run_audioldm2_audio2audio(
        pipe=pipe,
        mixed_waveform=mixed_wf,
        text_prompt=prompt,
        strength=0.5,
        num_inference_steps=200,
        guidance_scale=3.5,
        sr=sr,
        device=device
    )

    # 4. Save Final Audio as standard 16-bit PCM WAV
    out_path = os.path.join(output_folder, f"{video_id}_GEN.wav")
    # Clip between -1.0 and 1.0 to prevent audio clipping distortion
    generated_audio = np.clip(generated_audio, -1.0, 1.0)
    sf.write(out_path, generated_audio, sr, subtype="PCM_16")
    print(f"✅ Success! Audio file saved: {out_path}")