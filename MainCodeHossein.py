import os
import torch
import torchaudio
import torchaudio.transforms as T
import csv 
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
def run_score_driven_process(model, data_dict, description, shift=0):
    # Extract the first video key and its audio results
    video_key = list(data_dict.keys())[0]
    video_id = os.path.splitext(video_key)[0]
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
    raw_mix_path = os.path.join(output_dir, f"{video_id}_RAW_MIX.wav")
    torchaudio.save(raw_mix_path, synced_mix.cpu(), sr)
    print(f"📁 Raw mix saved for comparison: {raw_mix_path}")
    
    # 2. AI Naturalizer
    device = next(model.model.parameters()).device
    
    print("✨ Re-synthesizing into a unified soundscape...")
    seed = synced_mix.to(device)[..., :sr * 2]

    model.set_generation_params(duration=10.0, cfg_coeff=3.0)
    with torch.no_grad():
        output = model.generate_continuation(prompt=seed, 
                                             descriptions=[description], 
                                             prompt_sample_rate=sr,
                                             progress=True)
    
    # 3. Save and Preview

    final_path = os.path.join(output_dir, f"{video_id}_GEN.wav")
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
        return vgg_lookup.get(youtube_id, "natural environmental sound, high fidelity")

    results = [
            {'--XInAaMS6k.npy': {'-C6cbmMaENE.npy': 0.6466576988106167,
                        '-8KFpJHyspw.npy': 0.14904139426690938,
                        '-03N_1zOM4E.npy': 0.10659110957951061,
                        '-KQ7U3gS1wQ.npy': 0.05174921559957428,
                        '-HWoFxKmyyo.npy': 0.04596058174338897}}
    ,



    {'-0gYWIOfqdM.npy': {'-0gYWIOfqdM.npy': 0.6202382082358212,
                        '-4yCSY_5Zns.npy': 0.31029253698116976,
                        '-D7Od7iYq0A.npy': 0.04162212451171154,
                        '-A-xb-P-WxQ.npy': 0.02028793443981568,
                        '-HtBJbsbeHo.npy': 0.0075591958314815974}}
    ,



    {'-3M-k4nIYIM.npy': {'-9whJW7BUSU.npy': 0.3196014880596852,
                        '-HxQ9AoyRmY.npy': 0.24735819844202067,
                        '-60vY5Xw1qE.npy': 0.15602753270196487,
                        '-9vw5ZzChT0.npy': 0.14684911321636024,
                        '-3MNphBfq_0.npy': 0.13016366757996897}}
    ,



    {'-4ItJ9yTz_c.npy': {'-AioliAg12U.npy': 0.5865218525077042,
                        '-NPu34as_OY.npy': 0.18724029723388622,
                        '-6ZEGCtBKqs.npy': 0.10033209712998142,
                        '-Gbohom8C4Q.npy': 0.08084076879787816,
                        '-62pV95k9O0.npy': 0.045064984330550145}}
    ,



    {'-4o0jRbgHr4.npy': {'-NPu34as_OY.npy': 0.7716207865103134,
                        '-CexapzRAPQ.npy': 0.1304685089335117,
                        '-4o0jRbgHr4.npy': 0.05299553586319041,
                        '-9wRxzJ5j_Y.npy': 0.022736311823520462,
                        '-C8JU6yTJ40.npy': 0.022178856869464036}}
    ,



    {'-4rdRn-FRXo.npy': {'-Kc9P729mqM.npy': 0.36042281765518974,
                        '-7XYw1VrN64.npy': 0.33141335961690443,
                        '-MNP_aM09S8.npy': 0.1580765973593134,
                        '-CCbu3r-1pc.npy': 0.0915348425868341,
                        '-JdUSVmQq88.npy': 0.058552382781758214}}
    ,



    {'-6lkiUAf_cQ.npy': {'-6lkiUAf_cQ.npy': 0.7625829829950478,
                        '-ECRgvDx4xc.npy': 0.11362292998877925,
                        '-3xhrOw45ss.npy': 0.0643086250163537,
                        '-2JomCd5zzY.npy': 0.038541489457014064,
                        '-FfFD4bbCEI.npy': 0.020943972542805226}}
    ,



    {'-6VFTlZsft4.npy': {'-AltV1ftMk8.npy': 0.31224017893390565,
                        '-EWyYYBHsbQ.npy': 0.2170137781686798,
                        '-CcGuq0yoKo.npy': 0.20042663093106244,
                        '-0NxpZlO348.npy': 0.18478375634990274,
                        '-1EeNriiRN0.npy': 0.08553565561644937}}

    ]

# --- 3. LOAD MODEL ONCE ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🤖 Loading AudioGen-Medium into GPU memory...")
    shared_model = AudioGen.get_pretrained('facebook/audiogen-medium', device=device)

    # --- 4. EXECUTE LOOP ---
    for result in results:
        video_key = list(result.keys())[0]
        video_id = os.path.splitext(video_key)[0] 
        
        # Get the real label from our dictionary
        label = get_description(video_id)
        print(f"🎯 Using Label: '{label}' for Video: {video_id}")

        run_score_driven_process(
            model=shared_model,
            data_dict=result,
            description=label,
            shift=0 
        )
