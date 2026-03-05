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
    
    # 2. Setup SSD Cache (Important for your space management)
    # This ensures the 2GB model is stored on your SSD
    ssd_cache = "/media/hossein/H.s.wildwildwest/Bsc.Thesis_Datasets/torch_cache"
    os.makedirs(ssd_cache, exist_ok=True)
    os.environ['TORCH_HOME'] = ssd_cache
    
    # 3. Load Model
    print("\n🤖 Loading AudioGen-Medium...")
    print("(Note: If this is the first run, it will download ~1.5GB to your SSD)")
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    model.to(device)
    
    # 4. Generate a small sound
    description = "A soft wind blowing through forest leaves"
    print(f"✨ Generating: '{description}'...")
    
    # We generate only 3 seconds for a quick test
    model.set_generation_params(duration=3)
    
    with torch.no_grad():
        output = model.generate([description])
    
    # 5. Save to the current folder
    output_path = "test_generation.wav"
    torchaudio.save(output_path, output[0].cpu(), 16000)
    
    print(f"\n✅ SUCCESS! Generated file saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_test()