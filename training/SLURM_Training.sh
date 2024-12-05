#!/bin/bash
#SBATCH --partition=gpucluster
#SBATCH --qos=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=llama_vision_training
#SBATCH --gres=gpu:a100:4
#SBATCH --output=logs/training_job_%j.log

N=3  # Run the training 3 times sequentially
JOB_NUM=${1:-1}
EPOCHS_PER_JOB=1  # We'll split the 3 epochs across jobs
CHECKPOINT_ROOT="./button-detector"
WANDB_RUN_ID="llama_vision_$(date +%Y%m%d_%H%M%S)"

echo "Starting training job number $JOB_NUM at $(date)"
echo "Running on node: $(hostname)"

# Set up environment
source ~/git-repos/Visual-Data-Mining-AI-Model/venv_visual_data_mining/bin/activate

# Set up wandb run ID tracking
if [ $JOB_NUM -eq 1 ]; then
   mkdir -p "${CHECKPOINT_ROOT}"
   echo $WANDB_RUN_ID > "${CHECKPOINT_ROOT}/wandb_run_id.txt"
else
   WANDB_RUN_ID=$(cat "${CHECKPOINT_ROOT}/wandb_run_id.txt")
fi

# Function to check if previous run was successful
check_previous_run() {
   if [ $JOB_NUM -gt 1 ]; then
       if [ ! -d "$CHECKPOINT_ROOT" ]; then
           echo "Previous checkpoint not found at $CHECKPOINT_ROOT"
           exit 1
       else
           echo "Found valid checkpoint at $CHECKPOINT_ROOT"
       fi
   fi
}

# Function to run training
run_training() {
   # Calculate starting epoch based on job number
   START_EPOCH=$((($JOB_NUM - 1) * EPOCHS_PER_JOB))
   
   # Prepare checkpoint argument
   local checkpoint_arg=""
   if [ $JOB_NUM -gt 1 ]; then
       checkpoint_arg="--resume_from_checkpoint $CHECKPOINT_ROOT"
   fi

   # Run the training script
   python Main_finetune_Training.py \
       --checkpoint_dir $CHECKPOINT_ROOT \
       --start_epoch $START_EPOCH \
       --epochs_per_job $EPOCHS_PER_JOB \
       --wandb_run_id $WANDB_RUN_ID \
       $checkpoint_arg
}

# Check if we have previous checkpoints when needed
check_previous_run

# Run the training
run_training

# Check if training was successful
if [ $? -eq 0 ]; then
   echo "Training job $JOB_NUM completed successfully at $(date)"
   
   # Chain to next job if we haven't reached N
   if [ $JOB_NUM -lt $N ]; then
       NEXT_NUM=$((JOB_NUM + 1))
       echo "Submitting job number $NEXT_NUM"
       
       # Create a backup of the checkpoint before starting next job
       BACKUP_DIR="${CHECKPOINT_ROOT}_backup_job_${JOB_NUM}"
       echo "Creating backup at: $BACKUP_DIR"
       rm -rf "$BACKUP_DIR"  # Remove old backup if it exists
       cp -r $CHECKPOINT_ROOT "$BACKUP_DIR"
       
       # Submit the next job
       sbatch --dependency=afterok:$SLURM_JOB_ID $0 $NEXT_NUM
   else
       echo "All training jobs completed successfully!"
   fi
else
   echo "Training job $JOB_NUM failed at $(date)"
   exit 1
fi