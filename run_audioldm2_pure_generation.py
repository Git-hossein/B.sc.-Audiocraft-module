import argparse
import os
import json
import csv
import shutil
import soundfile as sf
import torch
from diffusers import AudioLDM2Pipeline
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

# --- MONKEY PATCH FOR TRANSFORMERS / AUDIOLDM2 ---
def _patch_update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
    model_kwargs["past_key_values"] = getattr(outputs, "past_key_values", None)
    if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
        mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [mask, mask.new_ones((mask.shape[0], 1))], dim=-1
        )
    return model_kwargs

GPT2Model._update_model_kwargs_for_generation = _patch_update_model_kwargs_for_generation


def run_audioldm2_text2audio(
    pipe,
    video_list,
    scratch_output_folder,
    descriptions=None,
    negative_prompt="low quality, distorted, noisy, glitch",
    num_inference_steps=100,
    guidance_scale=3.5,
    audio_length_in_s=10.0,
    seed=None,
    batch_size=2,
    sr=16000,
    device="cuda"
):
    """
    Batched Text-to-Audio generation for a list of video identifiers or filenames.
    """
    inference_queue = []

    for video_item in video_list:
        # Extract stem/vidID (e.g., 'abc.mp4' -> 'abc', or raw 'abc' -> 'abc')
        vid_id = os.path.splitext(os.path.basename(video_item))[0]
        
        prompt = None
        if descriptions and vid_id in descriptions:
            prompt = descriptions[vid_id]
        
        if prompt:
            inference_queue.append((vid_id, prompt))
        else:
            print(f"⚠️ Skipping item '{vid_id}': No prompt found in descriptions.")

    total_items = len(inference_queue)
    print(f"🚀 Processing {total_items} items in batches of {batch_size}...")

    for i in range(0, total_items, batch_size):
        batch = inference_queue[i : i + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_prompts = [item[1] for item in batch]
        batch_neg = [negative_prompt] * len(batch)

        # Set deterministic per-item generator if seed is supplied
        generator = None
        if seed is not None:
            generator = [
                torch.Generator(device=device).manual_seed(seed + i + idx)
                for idx in range(len(batch))
            ]

        print(f"[{i + 1}/{total_items}] Generating audio for: {batch_ids}")

        with torch.inference_mode():
            output = pipe(
                prompt=batch_prompts,
                negative_prompt=batch_neg,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=audio_length_in_s,
                generator=generator
            )

        audios = output.audios

        # Save generated audio as <vidID>_GEN.wav
        for vid_id, audio_waveform in zip(batch_ids, audios):
            out_filename = f"{vid_id}_GEN.wav"
            out_path = os.path.join(scratch_output_folder, out_filename)
            sf.write(out_path, audio_waveform, samplerate=sr)
            print(f"💾 Saved: {out_path}")


if __name__ == "__main__":
    # 1. SET UP ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="Parallel AudioLDM2 pure text2audio generation Worker")
    parser.add_argument("--session_id", type=str, required=True, help="The unique ID for this batch run")
    parser.add_argument("--chunk_id", type=int, required=True, help="The specific chunk index this node handles")
    args = parser.parse_args()

    # 2. CONSTRUCT PATHS
    TCML_INPUT_FOLDER = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input"
    session_dir = os.path.join(TCML_INPUT_FOLDER, args.session_id)
    chunk_file = os.path.join(session_dir, f"chunk_{args.chunk_id}.json")
    config_file = os.path.join(session_dir, "config.json")

    # 3. LOAD CONFIG & CHUNK
    print(f"📂 Task {args.chunk_id}: Loading session {args.session_id}")
    
    with open(config_file, "r") as f:
        conf = json.load(f)
    
    with open(chunk_file, "r") as f:
        chunk_data = json.load(f)

    # 4. SETUP ISOLATED SCRATCH OUTPUT DIRECTORY
    scratch_output_dir = os.environ.get('OUTPUT_DIR')
    if not scratch_output_dir:
        scratch_output_dir = f"/scratch/{os.environ.get('SLURM_JOB_ID', 'test')}/output"
    os.makedirs(scratch_output_dir, exist_ok=True)
    print(f"📂 Working output scratch directory: {scratch_output_dir}")

    # 5. LOAD DESCRIPTIONS (VGG-Sound)
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
        except FileNotFoundError as e:
            raise FileNotFoundError("⚠️ Error: 'vggsound.csv' not found. Cannot run without text descriptions.") from e
        return vgg_lookup

    vggsound_lookup = load_vggsound()

    # 6. LOAD PIPELINE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🤖 Loading AudioLDM 2 pipeline on {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    language_model = GPT2LMHeadModel.from_pretrained(
        "cvssp/audioldm2", 
        subfolder="language_model",
        torch_dtype=dtype
    )

    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", 
        language_model=language_model,
        torch_dtype=dtype
    ).to(device)

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        batch_size = 2 if ("1080 Ti" in gpu_name or "2080 Ti" in gpu_name) else 4
    else:
        gpu_name = "CPU"
        batch_size = 1

    print(f"⚡ GPU: {gpu_name} | Batch Size: {batch_size}")

    # 7. EXECUTE GENERATION
    run_audioldm2_text2audio(
        pipe=pipe,
        video_list=chunk_data,
        scratch_output_folder=scratch_output_dir,
        descriptions=vggsound_lookup,
        negative_prompt=conf.get("negative_prompt", "low quality, distorted, noisy, glitch"),
        num_inference_steps=conf.get("num_inference_steps", 100),
        guidance_scale=conf.get("guidance_scale", 3.5),
        audio_length_in_s=conf.get("audio_length_in_s", 10.0),
        seed=conf.get("seed", None),
        batch_size=batch_size,
        sr=16000,
        device=device
    )

    print(f"🏁 Task {args.chunk_id} completed successfully.")