import os
import sys
from typing import Literal
import torch
import torchaudio.transforms as T
import soundfile as sf
from diffusers import AudioLDM2Pipeline


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
    """
    audio_data: The dict of {filename: score}
    folder_path: Path to input waves
    """
    target_samples = 10 * sr
    final_mix = torch.zeros((1, target_samples))

    # We loop through the dictionary items directly
    for filename, scores in list(audio_data.items())[:num_audio_mix]:
        actual_wav_name = str(filename).replace('.npy', '.wav')
        path = os.path.join(folder_path, actual_wav_name)
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: {actual_wav_name} not found in folder. Skipping.")
            continue

        # Direct soundfile read (bypasses Torchaudio backend dispatcher completely)
        data, orig_sr = sf.read(path)
        wf = torch.from_numpy(data).float()
        
        # Format to 2D Tensor [channels, samples]
        if wf.ndim == 1:
            wf = wf.unsqueeze(0)
        else:
            wf = wf.T
        
        # Ensure mono channel
        if wf.shape[0] > 1:
            wf = torch.mean(wf, dim=0, keepdim=True)

        # Standardize Sample Rate
        if orig_sr != sr:
            print(f"🔄 Resampling {actual_wav_name}")
            wf = T.Resample(orig_sr, sr)(wf)
        
        # Ensure 10s length
        wf = wf[:, :target_samples]
        if wf.shape[1] < target_samples:
            wf = torch.nn.functional.pad(wf, (0, target_samples - wf.shape[1]))
            
        # Volume Balancing (RMS)
        energy = torch.sqrt(torch.mean(wf**2)) + 1e-8
        
        # Use the Similarity Score as the Weight
        score = scores[weight_by] if isinstance(scores, dict) else float(scores)
        balanced_wf = (wf / energy) * score
        
        final_mix += balanced_wf

    # Peak Normalization to make it audible and prevent clipping
    max_val = torch.max(torch.abs(final_mix))
    if max_val > 0:
        final_mix = final_mix / (max_val + 1e-8)

    return final_mix, sr


# --- MODULE: AUDIOLDM 2 INFERENCE ENGINE ---
def run_audioldm2_inference(
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
    Encodes mixed waveform into VAE latents, adds partial noise based on strength,
    and runs AudioLDM 2 diffusion.
    """
    # 1. Prepare raw audio tensor for VAE: shape [1, 1, samples]
    waveform = mixed_waveform.to(device=device, dtype=pipe.vae.dtype)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # 2. Encode mixed audio into initial VAE latents
    with torch.no_grad():
        inputs = pipe.feature_extractor(
            raw_speech=waveform.squeeze().cpu().numpy(), 
            sampling_rate=sr, 
            return_tensors="pt"
        )
        mel_spectrogram = inputs.input_features.to(device=device, dtype=pipe.vae.dtype)
        
        init_latents = pipe.vae.encode(mel_spectrogram).latent_dist.sample()
        init_latents = init_latents * pipe.vae.config.scaling_factor

    # 3. Calculate timesteps based on strength
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps_to_use = timesteps[t_start:]

    # 4. Add controlled noise to initial latents
    noise = torch.randn_like(init_latents)
    noisy_latents = pipe.scheduler.add_noise(init_latents, noise, timesteps_to_use[0:1])

    # 5. Run AudioLDM 2 generation
    with torch.no_grad():
        output = pipe(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            latents=noisy_latents,
            audio_length_in_s=10.0,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    
    return output.audios[0]


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # Example retrieval dictionary
    search_results = {'-jxXv_SoNaI.npy': {'13wjsR-VT_o.npy': {'softmax_score': 0.42002373933792114,
                                         'cosine_sim': 0.18692827224731445},
                     '0q53YxUMkKw.npy': {'softmax_score': 0.24088945984840393,
                                         'cosine_sim': 0.18136852979660034},
                     '-BAKe6QGTUk.npy': {'softmax_score': 0.13764320313930511,
                                         'cosine_sim': 0.1757718026638031},
                     '2Sr__ctC3s4.npy': {'softmax_score': 0.11249998956918716,
                                         'cosine_sim': 0.17375469207763672},
                     '1nZWM7d70Vk.npy': {'softmax_score': 0.08894357085227966,
                                         'cosine_sim': 0.17140518128871918}}}

    # Dynamic environment configuration
    input_folder = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
    output_folder = os.environ.get("OUTPUT_DIR", "./test_outputs")
    os.makedirs(output_folder, exist_ok=True)

    print(f"📂 Output directory set to: {output_folder}")

    # 1. Pick the query key and candidates
    query_key = list(search_results.keys())[0]
    candidates_dict = search_results[query_key]
    video_id = os.path.splitext(query_key)[0]

    print(f"🎬 Processing Query: {query_key}")

    # 2. Mix candidate audio files
    print("📊 Running intelligent_weighted_mix...")
    mixed_wf, sr = intelligent_weighted_mix(
        audio_data=candidates_dict,
        folder_path=input_folder,
        num_audio_mix=3,
        sr=48000,
        weight_by="softmax_score"
    )

    # Save RAW mix using soundfile directly
    raw_path = os.path.join(output_folder, f"{video_id}_RAW_MIX.wav")
    sf.write(raw_path, mixed_wf.squeeze().cpu().numpy(), sr)
    print(f"📁 Raw mix saved to scratch: {raw_path}")

    # 3. Load AudioLDM 2 Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Loading AudioLDM 2 pipeline on {device}...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # 4. Run AudioLDM 2 Generation
    prompt = "A high quality background sound matching the video scene"
    print(f"✨ Generating audio with prompt: '{prompt}'...")
    
    generated_audio = run_audioldm2_inference(
        pipe=pipe,
        mixed_waveform=mixed_wf,
        text_prompt=prompt,
        strength=0.5,
        num_inference_steps=200,
        guidance_scale=3.5,
        sr=sr,
        device=device
    )

    # 5. Save final generated wav result using soundfile directly
    out_path = os.path.join(output_folder, f"{video_id}_GEN.wav")
    sf.write(out_path, generated_audio, sr)
    print(f"✅ Success! Generated master file saved to scratch: {out_path}")