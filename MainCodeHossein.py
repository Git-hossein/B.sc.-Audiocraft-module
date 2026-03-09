import os
import torch
import torchaudio
import torchaudio.transforms as T
from audiocraft.models import AudioGen
#  Tell the 'transformers' library to stop complaining and just work
import transformers.utils.import_utils as import_utils
import_utils._torch_available = True 


# --- PATH & CACHE SETUP ---
# Ensure the model doesn't re-download every job
torch_cache = os.path.expanduser("~/.cache/torch_models")
os.makedirs(torch_cache, exist_ok=True)
os.environ['TORCH_HOME'] = torch_cache

def copy_input_to_scratch():
    slurm_job_id = os.environ.get('SLURM_JOB_ID') 

    if not slurm_job_id:
        raise RuntimeError("job not correctly started")
    
    scratch_path = f'/scratch/{slurm_job_id}/'
    os.makedirs(scratch_path, exist_ok=True)
    
    # Use f-string to make sure the path is correct
    os.system(f'cp -r /home/sherkat/B.sc.-Audiocraft-module/Hossein/input/* {scratch_path}') 
    return scratch_path

wav_input_folder_path = copy_input_to_scratch()


# --- MODULE 1: THE SCORE-BASED MIXER ---
def intelligent_weighted_mix(audio_data, folder_path, sr=16000):
    """
    audio_data: The dict of {filename: score}
    folder_path: Path to input waves
    """
    target_samples = 10 * sr
    final_mix = torch.zeros((1, target_samples))

    # We loop through the dictionary items directly
    for filename, score in audio_data.items():
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
        balanced_wf = (wf / energy) * score 
        
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
def run_score_driven_process(data_dict, description, shift=0):
    # Extract the first video key and its audio results
    video_key = list(data_dict.keys())[0]
    audio_results = data_dict[video_key]
    
    folder_path = wav_input_folder_path
    
    print(f"🎬 Processing Video: {video_key}")
    print(f"📊 Mixing {len(audio_results)} files based on Softmax scores...")

    # 1. Mix and Sync
    mixed_audio, sr = intelligent_weighted_mix(audio_results, folder_path)
    synced_mix = nudge_audio(mixed_audio, shift, sr)

    # --- SAVE THE RAW MIX FOR DEBUGGING ---
    output_dir = os.getenv('OUTPUT_DIR', '.')
    os.makedirs(output_dir, exist_ok=True)
    raw_mix_path = os.path.join(output_dir, "PRE_AI_RAW_MIX.wav")
    torchaudio.save(raw_mix_path, synced_mix.cpu(), sr)
    print(f"📁 Raw mix saved for comparison: {raw_mix_path}")
    
    # 2. AI Naturalizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🤖 Loading AudioGen on {device.upper()}...")
    
    model = AudioGen.get_pretrained('facebook/audiogen-medium', device=device)
    
    print("✨ Re-synthesizing into a unified soundscape...")
    seed = synced_mix.to(device)[..., :sr * 2]
    
    with torch.no_grad():
        output = model.generate_continuation(seed, [description], prompt_sample_rate=sr)
    
    # 3. Save and Preview

    final_path = os.path.join(output_dir, "SCORE_DRIVEN_MASTER.wav")
    torchaudio.save(final_path, output[0].cpu(), sr)
    
    print(f"✅ Success! Master file: {final_path}")
    return final_path

# --- DATA FROM YOUR SEARCH ---
search_results = {
    '-0gYWIOfqdM.npy': 
                  {'-0gYWIOfqdM.npy': 0.0011361405039085842,
                     '-4yCSY_5Zns.npy': 0.0011282989163786462,
                     '-D7Od7iYq0A.npy': 0.0011058588558776564,
                     '-A-xb-P-WxQ.npy': 0.001097940577780496,
                     '-HtBJbsbeHo.npy': 0.001087154364469816}}






# --- EXECUTE ---
final_master = run_score_driven_process(
    data_dict=search_results,
    description="A man plays a small, toy-like xylophone with two mallets",
    shift=0 # Change this to sync with video action
)
