import os
import sys
import torch
import torchaudio
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline

# Import your mixer from MainCodeHossein
from MainCodeHossein import intelligent_weighted_mix


# --- MODULE: AUDIOLDM 2 INFERENCE ENGINE ---
def run_audioldm2_inference(
    pipe: AudioLDM2Pipeline,
    mixed_waveform: torch.Tensor,
    text_prompt: str,
    negative_prompt: str = "low quality, distorted, noisy, glitch",
    strength: float = 0.5,
    num_inference_steps: int = 200,
    guidance_scale: float = 3.5,
    sr: int = 16000,
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
        sr=16000,
        weight_by="softmax_score"
    )

    # Save RAW mix for reference
    raw_path = os.path.join(output_folder, f"{video_id}_RAW_MIX.wav")
    torchaudio.save(raw_path, mixed_wf.cpu(), sr)
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

    # 5. Save final generated wav result
    out_path = os.path.join(output_folder, f"{video_id}_GEN.wav")
    scipy.io.wavfile.write(out_path, rate=sr, data=generated_audio)
    print(f"✅ Success! Generated master file saved to scratch: {out_path}")