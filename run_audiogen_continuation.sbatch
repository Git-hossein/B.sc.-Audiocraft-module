#!/bin/bash
#SBATCH --job-name=AudioGen_Array_Sherkat
#SBATCH --partition=day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:30:00
#SBATCH --array=0-0
#SBATCH --error=job.audiogen_continuation.%A_%a.err    # %A is Job ID, %a is Task Index
#SBATCH --output=job.audiogen_continuation.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL         # Only mail on failure to avoid spamming 32 emails
#SBATCH --mail-user=hossein.sherkat@student.uni-tuebingen.de

# --- 1. Define Paths ---
PROJECT_ROOT="/home/sherkat/B.sc.-Audiocraft-module"
# POINT TO YOUR NEW CUSTOM IMAGE HERE
IMAGE="/home/sherkat/singularity_build/audiogen_sandbox"
BASE_RESULT_DIR=~/audiogen_outputs

# --- 2. Setup Environment ---
SCRATCH_DIR="/scratch/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export INPUT_DIR="$SCRATCH_DIR/input"
export OUTPUT_DIR="$SCRATCH_DIR/output"
mkdir -p $OUTPUT_DIR


# Guaranteed cleanup on exit (normal, error, or killed)
trap 'echo "Removing scratch folder $SCRATCH_DIR"; rm -rf "$SCRATCH_DIR"' EXIT

# --- 3. Execute Script inside Singularity ---
# Removed 'source' because libraries are now INSIDE the .simg
singularity exec --nv --bind /scratch:/scratch $IMAGE bash -c "
    cd $PROJECT_ROOT && \
    nvidia-smi && \
    python3 -u run_audiogen_continuation.py \
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
if ls $OUTPUT_DIR/*_GEN.wav $OUTPUT_DIR/*_RAW_MIX.wav >/dev/null 2>&1; then
    
    mv -- "$OUTPUT_DIR"/*_GEN.wav "$TASK_DEST_DIR/"
    mv -- "$OUTPUT_DIR"/*_RAW_MIX.wav "$TASK_DEST_DIR/"
        
# Create a unique sentinel file for this specific task
    touch "$BASE_RESULT_DIR/${SESSION_ID}/DONE_${SLURM_ARRAY_TASK_ID}"
    echo "✅ Task $SLURM_ARRAY_TASK_ID Complete."
    
else
    echo "❌ Error: No audio files found for Task $SLURM_ARRAY_TASK_ID"
    exit 1
fi
