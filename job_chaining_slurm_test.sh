#!/bin/bash
# slurm_job.sh - Submit this to SLURM on SFSU

#SBATCH --partition=gpucluster
#SBATCH --qos=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=slurm_job_chaining_test
#SBATCH --output=chain_test_%j.log  # %j is the job ID

# Number of times to chain
N=2

JOB_NUM=${1:-1}
echo "Starting job number $JOB_NUM at $(date)"
echo "Running on node: $(hostname)"

# Run nvidia-smi every hour for three hours
for i in {1..3}; do
    echo "=== Check $i at $(date) ==="
    nvidia-smi
    sleep 3600
done

echo "Finished job $JOB_NUM at $(date)"

# Chain to next job if we haven't reached N
if [ $JOB_NUM -lt $N ]; then
    NEXT_NUM=$((JOB_NUM + 1))
    echo "Submitting job number $NEXT_NUM"
    sbatch --dependency=afterok:$SLURM_JOB_ID $0 $NEXT_NUM
fi
