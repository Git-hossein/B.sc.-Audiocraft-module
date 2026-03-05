import os
import torch
import torchaudio
from audiocraft.models import AudioGen

def run_test():
    # 1. Environment Check
    print("🔍 --- Environment Check ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Detected Device: {device.upper()}")
    print(f"Torch Version: {torch.__version__}")

    # 2. Setup Output Directory
    exported_dir = os.environ.get('OUTPUT_DIR')
    job_id = os.environ.get('SLURM_JOB_ID')

    if exported_dir:
        output_dir = exported_dir
        print(f"✅ Sync Match: Using exported OUTPUT_DIR: {output_dir}")
    elif job_id:
        output_dir = f"/scratch/{job_id}"
        print(f"⚠️ Warning: OUTPUT_DIR not found, falling back to manual scratch: {output_dir}")
    else:
        output_dir = "."
        print(f"ℹ️ Info: No Slurm environment detected, saving to current directory.")
    
    # 3. Load Model
    print("\n🤖 Loading AudioGen-Medium...")
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    model.to(device)
    
    # 4. Generate a small sound
    description = "A soft wind blowing through forest leaves"
    print(f"✨ Generating: '{description}'...")
    
    model.set_generation_params(duration=3)
    
    with torch.no_grad():
        output = model.generate([description])
    
    # 5. Save to the designated output folder (Scratch)
    output_path = os.path.join(output_dir, "test_generation.wav")
    torchaudio.save(output_path, output[0].cpu(), 16000)
    
    print(f"\n✅ SUCCESS! Generated file saved to: {output_path}")

if __name__ == "__main__":
    run_test()