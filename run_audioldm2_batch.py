import os
import sys
from typing import Literal, Optional
import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import csv
import argparse
import json
import shutil
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


# --- HELPER: ISOLATED SCRATCH DIRECTORY ---
def get_scratch_path(chunk_id: Optional[int] = None) -> str:
    """
    Returns an isolated scratch directory per SLURM job & array task
    to prevent collision when multiple tasks share a physical node.
    """
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "local_job")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or (str(chunk_id) if chunk_id is not None else "0")

    scratch_base = "/scratch" if os.path.exists("/scratch") else "./scratch"
    scratch_path = os.path.join(scratch_base, f"{job_id}_{task_id}")
    os.makedirs(scratch_path, exist_ok=True)
    return scratch_path


# --- HELPER: STAGE FILES TO NVME SCRATCH ---
def copy_input_to_scratch(inferred_dict: dict, scratch_path: str, source_folder: str):
    """
    Stages only candidate audio files needed by this chunk to local NVMe scratch.
    Skips missing files with a warning rather than crashing.
    """
    needed_files = set()
    for mix_list in inferred_dict.values():
        for filename in mix_list.keys():
            needed_files.add(str(filename).replace(".npy", ".wav"))

    copied, skipped = 0, 0
    for f in needed_files:
        src = os.path.join(source_folder, f)
        dst = os.path.join(scratch_path, f)

        if os.path.exists(dst):
            continue  # Already staged

        if os.path.exists(src):
            try:
                shutil.copy(src, dst)
                copied += 1
            except OSError as e:
                print(f"⚠️ Warning: Could not copy {f} to scratch: {e}")
        else:
            skipped += 1

    print(f"📦 Staged {copied} audio files to scratch ({skipped} not found in source).")


# --- MODULE 1: THE SCORE-BASED MIXER (16 kHz Native) ---
def intelligent_weighted_mix(
    audio_data: dict, 
    folder_path: str, 
    num_audio_mix: int = 3, 
    sr: int = 16000, 
    weight_by: Literal["softmax_score", "cosine_sim"] = "softmax_score"
):
    target_samples = 10 * sr  # Exactly 160,000 samples for AudioLDM 2 (10s @ 16kHz)
    final_mix = torch.zeros((1, target_samples))

    for filename, scores in list(audio_data.items())[:num_audio_mix]:
        actual_wav_name = str(filename).replace(".npy", ".wav")
        path = os.path.join(folder_path, actual_wav_name)
        
        if not os.path.exists(path):
            continue

        try:
            data, orig_sr = sf.read(path)
            wf = torch.from_numpy(data).float()
        except Exception as e:
            print(f"⚠️ Warning: Corrupt audio file {actual_wav_name}: {e}")
            continue
        
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
        weight = max(0.0, float(scores[weight_by]))
        balanced_wf = (wf / energy) * weight
        
        final_mix += balanced_wf

    final_mix = final_mix / (torch.max(torch.abs(final_mix)) + 1e-8)
    return final_mix, sr


# --- MODULE 2: 16kHz VAE MEL-SPECTROGRAM CONVERSION ---
def wav_to_vae_latents(
    waveform: torch.Tensor, 
    vae, 
    mel_transform: T.MelSpectrogram, 
    device: str = "cuda"
):
    """
    Computes the exact log-mel spectrogram (16kHz, 64 mel bins, hop 160)
    expected by AudioLDM / AudioLDM 2 VAE.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        waveform = waveform.squeeze(1)

    waveform = waveform.to(device)

    # 1. Extract Mel Spectrogram -> [batch, 64, time_steps]
    mel_spec = mel_transform(waveform)

    # 2. Log-compression
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    # 3. Shape to [batch, 1, time_steps, 64] for VAE encoder
    log_mel_spec = log_mel_spec.transpose(1, 2).unsqueeze(1).to(dtype=vae.dtype)

    # 4. VAE Encode
    init_latents = vae.encode(log_mel_spec).latent_dist.sample()
    init_latents = init_latents * vae.config.scaling_factor
    return init_latents


# --- MODULE 3: BATCHED AUDIO-TO-AUDIO DIFFUSION ---
def run_audioldm2_audio2audio(
    pipe: AudioLDM2Pipeline,
    inferred_dict: dict,
    scratch_input_folder: str,
    scratch_output_folder: str = "./test_outputs",
    descriptions: Optional[dict[str, str]] = None,
    negative_prompt: str = "low quality, distorted, noisy, glitch",
    strength: float = 0.45,
    num_inference_steps: int = 100,
    guidance_scale: float = 3.5,
    num_audio_mix: int = 3,
    weight_by: Literal["softmax_score", "cosine_sim"] = "softmax_score",
    batch_size: int = 4,
    sr: int = 16000,
    device: str = "cuda"
) -> list[str]:
    os.makedirs(scratch_output_folder, exist_ok=True)
    generated_files = []
    
    # Pre-allocate MelSpectrogram on GPU once
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
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

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)

    query_keys = list(inferred_dict.keys())
    total_queries = len(query_keys)

    # Process queries in mini-batches
    for i in range(0, total_queries, batch_size):
        batch_keys = query_keys[i : i + batch_size]
        curr_batch_size = len(batch_keys)
        
        print(f"\n📦 Processing Batch [{i + 1} - {i + curr_batch_size}/{total_queries}]")

        batch_waveforms = []
        batch_video_ids = []
        batch_prompts = []

        # 1. Mix Audio for each query in mini-batch
        for query_key in batch_keys:
            video_id = os.path.splitext(query_key)[0]
            candidates_dict = inferred_dict[query_key]
            batch_video_ids.append(video_id)

            mixed_wf, _ = intelligent_weighted_mix(
                audio_data=candidates_dict,
                folder_path=scratch_input_folder,
                num_audio_mix=num_audio_mix,
                sr=sr,
                weight_by=weight_by
            )
            batch_waveforms.append(mixed_wf.squeeze(0))

            # Save RAW Mix (16kHz PCM)
            raw_path = os.path.join(scratch_output_folder, f"{video_id}_RAW_MIX.wav")
            sf.write(raw_path, mixed_wf.squeeze().cpu().numpy(), sr, subtype="PCM_16")

            # Determine prompt
            if descriptions is not None:
                prompt_text = descriptions.get(video_id, descriptions.get(query_key, ""))
            else:
                prompt_text = ""
            batch_prompts.append(prompt_text)

        # Stack waveforms into [B, 160000]
        batch_audio_tensor = torch.stack(batch_waveforms).to(device)

        # 2. Text Conditioning Logic
        has_text = any(len(p.strip()) > 0 for p in batch_prompts)
        active_guidance = guidance_scale if (has_text and guidance_scale > 1.0) else 1.0

        # Reset scheduler for every mini-batch
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start_idx = max(num_inference_steps - init_timestep, 0)
        timesteps_to_use = timesteps[t_start_idx:]
        start_timestep = timesteps_to_use[0:1]

        with torch.no_grad():
            # 3. Encode Prompts
            prompt_embeds, prompt_mask, gen_prompt_embeds = pipe.encode_prompt(
                prompt=batch_prompts,
                device=device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=(active_guidance > 1.0),
                negative_prompt=[negative_prompt] * curr_batch_size if active_guidance > 1.0 else None,
            )

            # 4. Encode Batched 16kHz Audio into VAE Latents
            init_latents = wav_to_vae_latents(
                batch_audio_tensor, pipe.vae, mel_transform=mel_transform, device=device
            )

            # 5. Add Noise matching start_timestep
            noise = torch.randn_like(init_latents)
            latents = pipe.scheduler.add_noise(init_latents, noise, start_timestep)

            # 6. Batched Denoising Loop
            for t in timesteps_to_use:
                latent_model_input = torch.cat([latents] * 2) if active_guidance > 1.0 else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=gen_prompt_embeds,
                    encoder_hidden_states_1=prompt_embeds,
                    encoder_attention_mask_1=prompt_mask,
                    return_dict=False,
                )[0]

                if active_guidance > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + active_guidance * (noise_pred_text - noise_pred_uncond)

                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # 7. Decode Batched Latents -> Mel Spectrogram -> HiFi-GAN Vocoder
            latents = latents / pipe.vae.config.scaling_factor
            mel_spectrogram = pipe.vae.decode(latents).sample

            while mel_spectrogram.dim() > 3:
                mel_spectrogram = mel_spectrogram.squeeze(1)

            output_audio = pipe.vocoder(mel_spectrogram)  # Shape: [B, samples] or [B, 1, samples]
            output_audio = output_audio.cpu().float().numpy()

        # 8. Save Generated Master Waveforms
        target_len = int(10 * sr)
        for j, video_id in enumerate(batch_video_ids):
            out_path = os.path.join(scratch_output_folder, f"{video_id}_GEN.wav")
            audio_track = output_audio[j].squeeze()[:target_len]
            audio_track = np.clip(audio_track, -1.0, 1.0)
            sf.write(out_path, audio_track, sr, subtype="PCM_16")
            print(f"✅ [{video_id}] Saved: {out_path}")
            generated_files.append(out_path)

    return generated_files


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel AudioLDM2 batch runner")
    parser.add_argument("--session_id", type=str, required=True, help="Unique ID for this batch run")
    parser.add_argument("--chunk_id", type=int, required=True, help="Chunk index this node handles")
    args = parser.parse_args()

    TCML_INPUT_FOLDER = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input"
    session_dir = os.path.join(TCML_INPUT_FOLDER, args.session_id)
    chunk_file = os.path.join(session_dir, f"chunk_{args.chunk_id}.json")
    config_file = os.path.join(session_dir, "config.json")

    print(f"📂 Task {args.chunk_id}: Loading session {args.session_id}")
    
    with open(config_file, "r") as f:
        conf = json.load(f)
    
    with open(chunk_file, "r") as f:
        chunk_data = json.load(f)

    # 1. Setup isolated scratch directory for this task
    scratch_dir = os.environ.get("OUTPUT_DIR", get_scratch_path(chunk_id=args.chunk_id))
    os.makedirs(scratch_dir, exist_ok=True)
    print(f"📂 Working scratch directory: {scratch_dir}")

    # 2. Stage audio files to scratch
    copy_input_to_scratch(chunk_data, scratch_dir, source_folder=TCML_INPUT_FOLDER)

    # 3. Load descriptions if enabled
    def load_vggsound():
        print("📖 Loading VGGSound descriptions into memory...")
        vgg_lookup = {}
        csv_path = os.path.join(TCML_INPUT_FOLDER, "vggsound.csv")
        if not os.path.exists(csv_path):
            csv_path = "vggsound.csv"
        
        try:
            with open(csv_path, newline='', mode="r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        vgg_lookup[row[0].strip()] = row[2].strip()
        except FileNotFoundError:
            print("⚠️ Warning: 'vggsound.csv' not found. Running without text descriptions.")
        return vgg_lookup

    vggsound_lookup = load_vggsound() if conf.get("with_text_descr", True) else None

    # 4. Device and Batch Size Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Loading AudioLDM 2 pipeline on {device}...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        batch_size = 2 if ("1080 Ti" in gpu_name or "2080 Ti" in gpu_name) else 4
    else:
        batch_size = 1

    print(f"⚡ GPU: {gpu_name if device == 'cuda' else 'CPU'} | Batch Size: {batch_size}")

    # 5. Run Generation
    generated_audio = run_audioldm2_audio2audio(
        pipe=pipe,
        inferred_dict=chunk_data,
        scratch_input_folder=scratch_dir,
        scratch_output_folder=scratch_dir,
        descriptions=vggsound_lookup,
        negative_prompt=conf.get("negative_prompt", "low quality, distorted, noisy, glitch"),
        strength=conf.get("strength", 0.45),
        num_inference_steps=conf.get("num_inference_steps", 100),
        guidance_scale=conf.get("guidance_scale", 3.5),
        num_audio_mix=conf.get("num_audio_mix", 3),
        weight_by=conf.get("weight_by", "softmax_score"),
        batch_size=batch_size,
        sr=16000,
        device=device
    )
    print(f"🏁 Task {args.chunk_id} completed successfully.")