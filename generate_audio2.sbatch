#!/bin/bash
#SBATCH --job-name=AudioGen_Array_Sherkat
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:30:00
#SBATCH --array=0-0
#SBATCH --error=job.%A_%a.err    # %A is Job ID, %a is Task Index
#SBATCH --output=job.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL         # Only mail on failure to avoid spamming 32 emails
#SBATCH --mail-user=hossein.sherkat@student.uni-tuebingen.de

# --- 1. Define Paths ---
PROJECT_ROOT="/home/sherkat/B.sc.-Audiocraft-module"
# POINT TO YOUR NEW CUSTOM IMAGE HERE
IMAGE="/home/sherkat/singularity_build/audiogen_sandbox"
BASE_RESULT_DIR=~/audio_outputs

# --- 2. Setup Environment ---
export OUTPUT_DIR="/scratch/${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# --- 3. Execute Script inside Singularity ---
# Removed 'source' because libraries are now INSIDE the .simg
singularity exec --nv --bind /scratch:/scratch $IMAGE bash -c "
    cd $PROJECT_ROOT && \
    nvidia-smi && \
    python3 -u MainCodeHossein2.py \
            --session_id $SESSION_ID \
            --chunk_id $SLURM_ARRAY_TASK_ID
"

# --- 4. Move results back to a Task-Specific folder  ---
echo "Moving results from scratch to home..."
TASK_DEST_DIR="$BASE_RESULT_DIR/${SESSION_ID}/task_${SLURM_ARRAY_TASK_ID}"
readonly TASK_DEST_DIR
rm -rf "$TASK_DEST_DIR"
mkdir -p "$TASK_DEST_DIR"

# This prevents different tasks from overwriting each other's results
if ls $OUTPUT_DIR/*_GEN.wav >/dev/null 2>&1; then
    
    # Move the required GEN files
    mv -- "$OUTPUT_DIR"/*_GEN.wav "$TASK_DEST_DIR/"
    
    # 2. Check for and move OPTIONAL RAW_MIX files
    if ls $OUTPUT_DIR/*_RAW_MIX.wav >/dev/null 2>&1; then
        echo "Found optional RAW_MIX files, moving them..."
        mv -- "$OUTPUT_DIR"/*_RAW_MIX.wav "$TASK_DEST_DIR/"
    fi
        
    # Create the sentinel file (only if GEN files were successful)
    touch "$BASE_RESULT_DIR/${SESSION_ID}/DONE_${SLURM_ARRAY_TASK_ID}"
    echo "✅ Task $SLURM_ARRAY_TASK_ID Complete."
    
else
    # Fail if the core output is missing
    echo "❌ Error: Required *_GEN.wav files not found for Task $SLURM_ARRAY_TASK_ID"
    exit 1
fi
