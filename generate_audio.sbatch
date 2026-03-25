#!/bin/bash
#SBATCH --job-name=AudioGen_Sherkat
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.name@student.uni-tuebingen.de

# --- 1. Define Paths ---
PROJECT_ROOT="/home/sherkat/B.sc.-Audiocraft-module"
# POINT TO YOUR NEW CUSTOM IMAGE HERE
IMAGE="/home/sherkat/singularity_build/audiogen_sandbox"

# --- 2. Setup Environment ---
export OUTPUT_DIR="/scratch/$SLURM_JOB_ID"
mkdir -p $OUTPUT_DIR

# --- 3. Execute Script inside Singularity ---
# Removed 'source' because libraries are now INSIDE the .simg
singularity exec --nv $IMAGE bash -c "
    cd $PROJECT_ROOT && \
    nvidia-smi && \
    python3 -u MainCodeHossein.py
"

# --- 4. Move results back to Home ---
echo "Moving results from scratch to home..."
DEST_DIR=~/audio_outputs
readonly DEST_DIR
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

# Check if there are actually any .wav files before moving
# We use -- to signal the end of command options in case files start with -
if ls $OUTPUT_DIR/*_GEN.wav $OUTPUT_DIR/*_RAW_MIX.wav >/dev/null 2>&1; then
    
    # Move files AND check both operations
    if mv -- "$OUTPUT_DIR"/*_GEN.wav "$DEST_DIR/" \
       && mv -- "$OUTPUT_DIR"/*_RAW_MIX.wav "$DEST_DIR/"; then
        
        touch "$DEST_DIR/JOB_${SLURM_JOB_ID}_DONE"
        echo "... Sentinel file created..."

        echo "✅ Success! Raw mix and Master mix moved to $DEST_DIR"
        ls -lh "$DEST_DIR"
    
    else
        echo "⚠️ Move failed, not creating sentinel."
    fi
else
    echo "❌ Error: No matching audio files (_GEN.wav or _RAW_MIX.wav) found in scratch."
fi
