#!/bin/bash
#SBATCH --job-name=AudioLDM2_T2A_Sherkat
#SBATCH --partition=day
#SBATCH --gres=gpu:1          # Requesting 2080 Ti
#SBATCH --nodelist=tcml-node37
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:30:00
#SBATCH --array=0-0                 # Set to your total chunks, e.g., 0-19 for 20 chunks
#SBATCH --error=job.AudioLDM2_T2A.%A_%a.err        # %A is Job ID, %a is Task Index
#SBATCH --output=job.AudioLDM2_T2A.%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hossein.sherkat@student.uni-tuebingen.de

# --- 1. Define Session & Paths ---
SESSION_ID="${SESSION_ID:-test_session}"
PROJECT_ROOT="/home/sherkat/B.sc.-Audiocraft-module"
IMAGE="/home/sherkat/singularity_audioLDM/audioldm2.sif"
BASE_RESULT_DIR="$HOME/audioLDM2_generation_outputs"

# --- 2. Setup Scratch & Cache Environments ---
SCRATCH_DIR="/scratch/${SLURM_JOB_ID}"
echo "my scratch id is $SLURM_JOB_ID"
export OUTPUT_DIR="$SCRATCH_DIR/output"
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch_models"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"

# Guaranteed cleanup on exit (normal, error, or killed)
trap 'echo "Removing contents of scratch folder $SCRATCH_DIR"; rm -rf "${SCRATCH_DIR:?}/"*' EXIT

# --- 3. Execute Script inside Singularity Container ---
singularity exec --nv \
    --bind /scratch:/scratch \
    --bind "$HOME:$HOME" \
    "$IMAGE" bash -c "
        cd $PROJECT_ROOT && \
        nvidia-smi && \
        python3 -u run_audioldm2_text2audio_batch.py \
            --session_id '$SESSION_ID' \
            --chunk_id '$SLURM_ARRAY_TASK_ID'
"

# --- 4. Move Results from Scratch to Home Storage ---
echo "Moving results from local scratch to home storage..."
TASK_DEST_DIR="$BASE_RESULT_DIR/${SESSION_ID}/task_${SLURM_ARRAY_TASK_ID}"
readonly TASK_DEST_DIR
rm -rf "$TASK_DEST_DIR"
mkdir -p "$TASK_DEST_DIR"

if ls "$OUTPUT_DIR"/*_GEN.wav >/dev/null 2>&1; then
    mv -- "$OUTPUT_DIR"/*_GEN.wav "$TASK_DEST_DIR/"
        
    # Create a unique sentinel file for this specific task
    touch "$BASE_RESULT_DIR/${SESSION_ID}/DONE_${SLURM_ARRAY_TASK_ID}"
    echo "✅ Task $SLURM_ARRAY_TASK_ID Complete. Saved to: $TASK_DEST_DIR"
else
    echo "❌ Error: No generated audio files found in $OUTPUT_DIR for Task $SLURM_ARRAY_TASK_ID"
    exit 1
fi