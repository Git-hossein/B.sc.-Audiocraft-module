import os
import torch
import torchaudio
import torchaudio.transforms as T
import csv 
from audiocraft.models import AudioGen
#  Tell the 'transformers' library to stop complaining and just work
import transformers.utils.import_utils as import_utils
import_utils._torch_available = True 
import warnings
import json
from typing import Optional, Any, Literal
import shutil


# --- PATH & CACHE SETUP ---
# Ensure the model doesn't re-download every job
torch_cache = os.path.expanduser("~/.cache/torch_models")
os.makedirs(torch_cache, exist_ok=True)
os.environ['TORCH_HOME'] = torch_cache


def get_scratch_path():
    slurm_job_id = os.environ.get('SLURM_JOB_ID') 

    if not slurm_job_id:
        raise RuntimeError("job not correctly started")

    scratch_path = f'/scratch/{slurm_job_id}/'
    os.makedirs(scratch_path, exist_ok=True)

    return scratch_path

wav_input_folder_path = get_scratch_path()

def copy_input_to_scratch(inferred_dict, source_folder = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"):

    needed_files = set()
    for mix_list in inferred_dict.values():
        for filename in mix_list.keys():
            needed_files.add(filename.replace('.npy', '.wav'))

    scratch_path = wav_input_folder_path
    
    # Use f-string to make sure the path is correct
    for f in needed_files:
        try:
            shutil.copy(os.path.join(source_folder, f), scratch_path)
        except Exception as e:
            print(f"something went wrong while copying to scratch: {e}")



# --- MODULE 1: THE SCORE-BASED MIXER ---
def intelligent_weighted_mix(audio_data, folder_path, num_audio_mix, sr=16000, weight_by: Literal["softmax_score", "cosine_sim"] = "softmax_score"):
    """
    audio_data: The dict of {filename: score}
    folder_path: Path to input waves
    """
    target_samples = 10 * sr
    final_mix = torch.zeros((1, target_samples))

    # We loop through the dictionary items directly
    for filename, scores in list(audio_data.items())[:num_audio_mix]:
        # Your dict has .npy, but the folder has .wav
        # We replace the extension to find the actual audio file
        actual_wav_name = str(filename).replace('.npy', '.wav')
        path = os.path.join(folder_path, actual_wav_name)
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: {actual_wav_name} not found in folder. Skipping.")
            continue

        wf, orig_sr = torchaudio.load(path)
        
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
        # We multiply by a 'boost' factor (e.g., 100) if scores are very small
        balanced_wf = (wf / energy) * scores[weight_by]
        
        final_mix += balanced_wf

    # Peak Normalization to make it audible and prevent clipping
    final_mix = final_mix / (torch.max(torch.abs(final_mix)) + 1e-8)
    return final_mix, sr

# --- MODULE 2: THE ALIGNER ---
def nudge_audio(waveform, shift_seconds, sr):
    samples_to_shift = int(shift_seconds * sr)
    total_len = waveform.shape[1]
    
    if samples_to_shift > 0: 
        silence = torch.zeros((waveform.shape[0], samples_to_shift))
        shifted = torch.cat([silence, waveform], dim=1)
        return shifted[:, :total_len] 
    else: 
        shifted = waveform[:, abs(samples_to_shift):]
        return torch.nn.functional.pad(shifted, (0, total_len - shifted.shape[1]))

# --- MODULE 3: THE MASTER EXECUTION ---
def run_score_driven_process(
        infered_dict: dict[str, dict[str, dict[str, float]]], 
        descriptions: list[str], 
        weight_by: Literal["softmax_score", "cosine_sim"] = "softmax_score",
        cfg_coef = 3.0, 
        prompt_duration = 2, 
        num_audio_mix: Optional[int] = None, 
        shift: int = 0, 
        shared_model: Optional[AudioGen] = None,
        batch_size = 4
    )-> list[str]:
    """
    Processes a batch of video audio results by mixing them and re-synthesizing 
    using AudioGen.

    Args:
        data_dict (Dict[str, Dict[str, float]]): A nested dictionary where keys are 
            video filenames and values are dictionaries of inferred audio 
            segments with their corresponding softmax scores.
        descriptions (List[str]): A list of text descriptions/prompts for each 
            video, ordered to match data_dict.keys().
        shift (int, optional): Number of samples to shift/nudge the mixed audio 
            for synchronization. Defaults to 0.
        shared_model (Optional[AudioGen], optional): A pre-loaded AudioGen model 
            instance. If None, the model is loaded locally. Defaults to None.

    Returns:
        List[str]: A list of file paths to the successfully generated audio files.

    Raises:
        TypeError: If shared_model is provided but is not an AudioGen instance.
        ValueError: If the number of descriptions does not match the number of videos.
    """

    assert len(set([len(score) for score in infered_dict.values()])) == 1, "this inference dict is not homogounes!"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if shared_model is not None:
        if not isinstance(shared_model, AudioGen):
            raise TypeError(
                f"❌ Invalid shared_model type: {type(shared_model)}. "
                f"This function specifically requires audiogen-medium model to "
                f"handle continuation prompts."
            )
        model = shared_model
        print("Using verified shared AudioGen model...")
    else:
        warnings.warn("⚠️ No shared_model provided. Loading AudioGen-Medium locally.")
        model = AudioGen.get_pretrained('facebook/audiogen-medium', device=device)

    generated_files = []

    if len(descriptions) != len(infered_dict):
        raise ValueError(f"❌ Mismatch: {len(infered_dict)} videos but {len(descriptions)} descriptions.")
    

    inferred_dict_keys = list(infered_dict.keys())
    for i in range(0, len(inferred_dict_keys), batch_size):

        batch_keys = inferred_dict_keys[i: i+batch_size]
        batch_descriptions = descriptions[i: i+batch_size]
        batch_seeds = []
        batch_video_ids = []
        folder_path = wav_input_folder_path

        for video in batch_keys:
            # Extract the video id and its audio results
            video_id = os.path.splitext(video)[0]
            batch_video_ids.append(video_id)
            audio_results = infered_dict[video]
            print(f"🎬 Processing Video: {video}")
            print(f"📊 Mixing {len(audio_results)} files based on Softmax scores...")
            
            # 1. Mix and Sync
            k = min(len(audio_results), num_audio_mix) if num_audio_mix is not None else len(audio_results)
            mixed_audio, sr = intelligent_weighted_mix(audio_results, folder_path, num_audio_mix = k ,weight_by= weight_by)
            synced_mix = nudge_audio(mixed_audio, shift, sr)
            batch_seeds.append(synced_mix)

            # --- SAVE THE RAW MIX FOR DEBUGGING ---
            output_dir = os.getenv('OUTPUT_DIR', '.')
            os.makedirs(output_dir, exist_ok=True)
            raw_mix_path = os.path.join(output_dir, f"{video_id}_RAW_MIX.wav")
            torchaudio.save(raw_mix_path, synced_mix.cpu(), sr)
            print(f"📁 Raw mix saved for comparison: {raw_mix_path}")
                
        seeds_tensor = torch.stack(batch_seeds).to(device)
        # 2. AI Naturalizer

        print(f"🤖 Processing on: {device.upper()}")
        
        print("✨ Re-synthesizing into a unified soundscape...")
        seeds_tensor = seeds_tensor[..., :sr * prompt_duration]
        active_cfg = cfg_coef if any(d != None for d in batch_descriptions) else 0.0
        model.set_generation_params(duration=10.0, cfg_coef= active_cfg)
        with torch.no_grad():
            output_batch = model.generate_continuation(prompt=seeds_tensor, 
                                                descriptions=batch_descriptions, 
                                                prompt_sample_rate=sr,
                                                progress=True)
        
        # 3. Save and Preview
        for j, output_audio in enumerate(output_batch):
                    video_id = batch_video_ids[j]
                    final_path = os.path.join(output_dir, f"{video_id}_GEN.wav")
                    
                    # output_audio is already the specific tensor for this video
                    torchaudio.save(final_path, output_audio.cpu(), sr)
                    
                    print(f"✅ Success! Master file: {final_path}")
                    generated_files.append(final_path)

            # --- CLEANUP CACHE AFTER 5 BATCHES---
        if torch.cuda.is_available()and (i // batch_size + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return generated_files

# inferred dict example
# search_results = {
#     '-0gYWIOfqdM.npy': 
#                   {'-0gYWIOfqdM.npy': {'softmax_score':0.32, 'cosine_sim': 0.01},
#                      '-4yCSY_5Zns.npy': {'softmax_score':0.12, 'cosine_sim': 0.005},
#                      '-D7Od7iYq0A.npy': {'softmax_score':0.3222, 'cosine_sim': 0.03},
#                      '-A-xb-P-WxQ.npy': {'softmax_score':0.001, 'cosine_sim': 0.012},
#                      '-HtBJbsbeHo.npy': {'softmax_score':0.0201, 'cosine_sim': 0.0111}}}




if __name__ == "__main__":

# --- 1. PRE-LOAD THE DESCRIPTIONS ---
    print("📖 Loading VGGSound descriptions into memory...")
    vgg_lookup = {}
    try:
        with open("vggsound.csv", newline='', mode="r") as f: # Ensure the filename is correct
            reader = csv.reader(f)
            for row in reader:
                # row[0] = ID, row[2] = Label, row[3] = train/test
                if row[3] == "train":
                    vgg_lookup[row[0].strip()] = row[2].strip()
    except FileNotFoundError:
        print("❌ Error: 'vggsound.csv' not found. Check your PROJECT_ROOT.")
        vgg_lookup = {}

    def get_description(youtube_id):
        # Look up the ID; if not found, use a safe default
        return vgg_lookup.get(youtube_id, None)

    # if u wanna use the json file: 
    audio_input_folder = "/home/sherkat/B.sc.-Audiocraft-module/Hossein/input/"
    inferred_dict_json_file = "inferred.json"
    with open(os.path.join(audio_input_folder, inferred_dict_json_file) , "r") as f:
        results = json.load(f)

    copy_input_to_scratch(results)
    # --- 3. LOAD MODEL ONCE ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🤖 Loading AudioGen-Medium into GPU memory...")
    shared_model = AudioGen.get_pretrained('facebook/audiogen-medium', device=device)

    # --- 4. EXECUTE LOOP ---
    labels = []
    for video in results.keys():
        video_id = os.path.splitext(video)[0] 
        label = get_description(video_id)
        labels.append(label)

    run_score_driven_process(
        infered_dict=results,
        descriptions=labels,
        weight_by= "softmax_score",
        cfg_coef = 3.0, 
        prompt_duration = 2, 
        num_audio_mix = 5, 
        shift=0,
        shared_model=shared_model
    )
