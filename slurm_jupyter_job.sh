#!/bin/bash
# slurm_jupyter_job.sh - Submit this to SLURM on SFSU

#SBATCH --partition=gpucluster
#SBATCH --qos=interactive
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=slurm_jupyter
#SBATCH --output=jupyter_%j.log  # %j is the job ID

# Activate environment
source ~/git-repos/Visual-Data-Mining-AI-Model/venv_visual_data_mining/bin/activate

# Find a free port
PORT=$(python3 -c '
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
')

# Save connection info for the client
echo $PORT > ~/.jupyter_port
echo $(hostname) > ~/.jupyter_node

# Start Jupyter and save the token
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0 2>&1 | tee ~/.jupyter_output
