#!/bin/bash
# slurm_job.sh - Submit this to SLURM on SFSU
#SBATCH --partition=gpucluster
#SBATCH --qos=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=llama_training
#SBATCH --output=training_job_%j.log

# Number of jobs to chain
N=3  # This will run the training 3 times sequentially
JOB_NUM=${1:-1}
EPOCHS_PER_JOB=3  # Number of epochs per job

echo "Starting training job number $JOB_NUM at $(date)"
echo "Running on node: $(hostname)"

# Set up environment
source ~/git-repos/Visual-Data-Mining-AI-Model/venv_visual_data_mining/bin/activate

# Function to check if previous run was successful
check_previous_run() {
    if [ $JOB_NUM -gt 1 ]; then
        if [ ! -d "./finetuned_model/fine-tuned" ]; then
            echo "Previous checkpoint not found! Exiting."
            exit 1
        fi
    fi
}

# Function to run training
run_training() {
    local resume_flag=""
    if [ $JOB_NUM -gt 1 ]; then
        resume_flag="--resume_from_checkpoint ./finetuned_model/fine-tuned"
    fi

    torchrun --nnodes 1 --nproc_per_node 4 finetuning.py \
        --enable_fsdp \
        --lr 1e-5 \
        --num_epochs $EPOCHS_PER_JOB \
        --batch_size_training 8 \
        --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
        --dist_checkpoint_root_folder ./finetuned_model \
        --dist_checkpoint_folder fine-tuned \
        --use_fast_kernels \
        --dataset "custom_dataset" \
        --custom_dataset.test_split "test" \
        --custom_dataset.file "web_scraper_dataset.py" \
        --run_validation True \
        --batching_strategy padding \
        --use_peft \
        --peft_method lora \
        $resume_flag
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
        cp -r ./finetuned_model/fine-tuned "./finetuned_model/fine-tuned_backup_job_${JOB_NUM}"
        
        # Submit the next job
        sbatch --dependency=afterok:$SLURM_JOB_ID $0 $NEXT_NUM
    else
        echo "All training jobs completed successfully!"
    fi
else
    echo "Training job $JOB_NUM failed at $(date)"
    exit 1
fi