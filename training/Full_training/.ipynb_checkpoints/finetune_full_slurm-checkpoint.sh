#!/bin/bash
#SBATCH --partition=gpucluster
#SBATCH --qos=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=llama_training
#SBATCH --output=logs/training_job_%j.log

N=1  # This will run the training 3 times sequentially
JOB_NUM=${1:-1}
EPOCHS_PER_JOB=3  # Number of epochs per job
CHECKPOINT_ROOT="./finetuned_model"
CHECKPOINT_DIR="fine-tuned"
FULL_CHECKPOINT_PATH="${CHECKPOINT_ROOT}/${CHECKPOINT_DIR}"

echo "Starting training job number $JOB_NUM at $(date)"
echo "Running on node: $(hostname)"

# Set up environment
source ~/git-repos/Visual-Data-Mining-AI-Model/venv_visual_data_mining/bin/activate

# Set up wandb run ID tracking
if [ $JOB_NUM -eq 1 ]; then
   # For first job, create a new run ID
   WANDB_RUN_ID=$(wandb run id 2>/dev/null || date +%Y%m%d_%H%M%S)
   mkdir -p "${CHECKPOINT_ROOT}"
   echo $WANDB_RUN_ID > "${CHECKPOINT_ROOT}/wandb_id.txt"  # Changed from training_run_id.txt
else
   # For continuation jobs, read the existing run ID
   WANDB_RUN_ID=$(cat "${CHECKPOINT_ROOT}/wandb_id.txt")   # Changed from training_run_id.txt
fi

# Function to check if previous run was successful
check_previous_run() {
   if [ $JOB_NUM -gt 1 ]; then
       # Check for the specific files that should exist in a complete checkpoint
       if [ ! -d "$FULL_CHECKPOINT_PATH" ]; then
           echo "Previous checkpoint not found at $FULL_CHECKPOINT_PATH"
           echo "Contents of checkpoint directory (if it exists):"
           ls -la $CHECKPOINT_ROOT || echo "Directory does not exist"
           exit 1
       else
           echo "Found valid checkpoint at $FULL_CHECKPOINT_PATH"
       fi
   fi
}

# Function to run training
run_training() {
   local checkpoint_flag=""
   if [ $JOB_NUM -gt 1 ]; then
       echo "Loading from previous checkpoint: $FULL_CHECKPOINT_PATH"
   else
       echo "Starting fresh training"
       # Preserve wandb ID if it exists
       if [ -f "$FULL_CHECKPOINT_PATH/wandb_id.txt" ]; then
           mkdir -p /tmp/wandb_backup
           cp "$FULL_CHECKPOINT_PATH/wandb_id.txt" /tmp/wandb_backup/
       fi
       rm -rf $FULL_CHECKPOINT_PATH
       mkdir -p $FULL_CHECKPOINT_PATH
       # Restore wandb ID if it was backed up
       if [ -f "/tmp/wandb_backup/wandb_id.txt" ]; then
           cp /tmp/wandb_backup/wandb_id.txt "$FULL_CHECKPOINT_PATH/"
           rm -rf /tmp/wandb_backup
       fi
   fi

   # Print debug information
   echo "Current directory: $(pwd)"
   echo "Checkpoint directory structure:"
   ls -R $CHECKPOINT_ROOT
   echo "Using training run ID: $TRAINING_RUN_ID"

   # Use a random port to avoid conflicts
   RANDOM_PORT=$(shuf -i 29500-65000 -n 1)
   echo "Using random port: $RANDOM_PORT"

   # Run the training with the working configuration
   torchrun --nnodes 1 --nproc_per_node 4 --master_port $RANDOM_PORT finetuning.py \
       --enable_fsdp \
       --lr 1e-5 \
       --num_epochs $EPOCHS_PER_JOB \
       --batch_size_training 8 \
       --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
       --dist_checkpoint_root_folder $CHECKPOINT_ROOT \
       --dist_checkpoint_folder $CHECKPOINT_DIR \
       --use_fast_kernels \
       --dataset "custom_dataset" \
       --custom_dataset.test_split "test" \
       --custom_dataset.file "web_scraper_dataset.py" \
       --run_validation True \
       --batching_strategy padding \
       --output_dir "$FULL_CHECKPOINT_PATH" \
       --use_wandb True \
       --wandb_config.project "llama_recipes" \
       --wandb_config.group "$TRAINING_RUN_ID"
}

# Create checkpoint root directory if it doesn't exist
mkdir -p $CHECKPOINT_ROOT

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
       BACKUP_DIR="${CHECKPOINT_ROOT}/fine-tuned_backup_job_${JOB_NUM}"
       echo "Creating backup at: $BACKUP_DIR"
       rm -rf "$BACKUP_DIR"  # Remove old backup if it exists
       cp -r $FULL_CHECKPOINT_PATH "$BACKUP_DIR"
       
       # Submit the next job
       sbatch --dependency=afterok:$SLURM_JOB_ID $0 $NEXT_NUM
   else
       echo "All training jobs completed successfully!"
   fi
else
   echo "Training job $JOB_NUM failed at $(date)"
   echo "Final checkpoint directory contents:"
   ls -R $CHECKPOINT_ROOT
   exit 1
fi