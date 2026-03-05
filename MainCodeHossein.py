import os
import sys

# --- FORCE FIX START ---
# 1. Hide TensorFlow so it stops bullying PyTorch
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
sys.modules['tensorflow'] = None

# 2. Tell the 'transformers' library to stop complaining and just work
import transformers.utils.import_utils as import_utils
import_utils._torch_available = True 
# --- FORCE FIX END ---

import torch
import torchaudio
from audiocraft.models import AudioGen
import traceback

def run_test():
    print("🔍 --- Environment Check ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Detected Device: {device.upper()}")
    print(f"Torch Version: {torch.__version__}")
    
    # Setup Cache
    torch_cache = os.path.expanduser("~/.cache/torch_models")
    os.makedirs(torch_cache, exist_ok=True)
    os.environ['TORCH_HOME'] = torch_cache

    output_dir = os.getenv('OUTPUT_DIR', '.')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\n🤖 Loading AudioGen-Medium...")
        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        model.to(device)
        
        description = "A soft wind blowing through forest leaves"
        print(f"✨ Generating: '{description}'...")
        
        model.set_generation_params(duration=3)
        with torch.no_grad():
            output = model.generate([description])
        
        output_path = os.path.join(output_dir, "test_generation.wav")
        torchaudio.save(output_path, output[0].cpu(), 16000)
        
        print(f"\n✅ SUCCESS! File saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()