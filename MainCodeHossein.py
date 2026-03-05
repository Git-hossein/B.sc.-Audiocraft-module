import os
import torch
import torchaudio
from audiocraft.models import AudioGen
import traceback

def run_test():
    # 1. Environment Check
    print("🔍 --- Environment Check ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Detected Device: {device.upper()}")
    print(f"Torch Version: {torch.__version__}")
    
    # 2. Setup Safe Cluster Cache (Home Directory)
    # We use ~/.cache because you have write-access there.
    torch_cache = os.path.expanduser("~/.cache/torch_models")
    os.makedirs(torch_cache, exist_ok=True)
    os.environ['TORCH_HOME'] = torch_cache
    print(f"📂 Model Cache: {torch_cache}")

    # Fetch Slurm variables
    output_dir = os.getenv('OUTPUT_DIR', '.')
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Load Model and Generate (with Error Catching)
    try:
        print("\n🤖 Loading AudioGen-Medium...")
        # Note: This might take a while to download 1.5GB
        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        model.to(device)
        
        description = "A soft wind blowing through forest leaves"
        print(f"✨ Generating: '{description}'...")
        
        model.set_generation_params(duration=3)
        
        with torch.no_grad():
            output = model.generate([description])
        
        # 5. Save to the designated output folder
        output_path = os.path.join(output_dir, "test_generation.wav")
        # Ensure tensor is on CPU and correct shape for saving
        audio_data = output[0].cpu()
        torchaudio.save(output_path, audio_data, 16000)
        
        print(f"\n✅ SUCCESS! File saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ PYTHON CRASHED during model phase: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()