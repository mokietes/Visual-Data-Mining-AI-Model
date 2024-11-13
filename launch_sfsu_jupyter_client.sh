#!/bin/bash
# launch_jupyter.sh - Run this from your local machine

# Function to check if job is running
check_job_status() {
    local job_id=$1
    local status=$(ssh sfsu "squeue -j $job_id -h -o %t" 2>/dev/null)
    echo $status
}

echo "Submitting Jupyter job to SLURM..."
JOB_ID=$(ssh sfsu "sbatch ~/git-repos/Visual-Data-Mining-AI-Model/slurm_jupyter_job.sh" | awk '{print $4}') || exit 1
echo "Submitted job ID: $JOB_ID"
echo "Waiting for job to start..."

# Wait for job to be running
while true; do
    status=$(check_job_status $JOB_ID)
    if [ "$status" == "R" ]; then
        echo "Job is running!"
        break
    elif [ -z "$status" ]; then
        echo "Job failed to start. Check jupyter_${JOB_ID}.log on SFSU"
        exit 1
    fi
    echo "Job status: $status (waiting for R)"
    sleep 3
done

# Give Jupyter a moment to write its files
sleep 5
echo "Getting connection details..."

# Get node and port from log file, with better parsing
while true; do
    LOG_LINE=$(ssh sfsu "grep -m1 'http://' jupyter_${JOB_ID}.log 2>/dev/null | grep -v 127.0.0.1")
    if [ ! -z "$LOG_LINE" ]; then
        NODE=$(echo "$LOG_LINE" | sed -n 's|.*http://\([^:]*\):\([0-9]*\)/.*|\1|p')
        PORT=$(echo "$LOG_LINE" | sed -n 's|.*http://[^:]*:\([0-9]*\)/.*|\1|p')
        if [ ! -z "$NODE" ] && [ ! -z "$PORT" ]; then
            break
        fi
    fi
    echo "Waiting for Jupyter to initialize..."
    sleep 2
done

# Get the token from the log with better parsing
TOKEN=$(ssh sfsu "grep -m1 'token=' jupyter_${JOB_ID}.log | sed -n 's/.*token=\([^&]*\).*/\1/p'")

echo -e "\nJupyter is running!"
echo "Node: $NODE"
echo "Port: $PORT"

# Display both local and remote URLs
echo -e "\nConnection URLs:"
echo "Local  URL: http://localhost:$PORT/?token=$TOKEN"
echo "Remote URL: http://$NODE:$PORT/?token=$TOKEN"

echo -e "\nJob Management:"
echo "Check status: ssh sfsu \"squeue -j $JOB_ID\""
echo "View logs:    ssh sfsu \"cat jupyter_${JOB_ID}.log\""
echo "Kill job:     ssh sfsu \"scancel $JOB_ID\""

echo -e "\nEstablishing SSH tunnel (keep this terminal open)..."

# Try to establish SSH tunnel with retry
max_retries=5
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if ssh -N -L "$PORT:$NODE:$PORT" sfsu; then
        break
    fi
    if [ $? -eq 130 ]; then  # Ctrl+C was pressed
        break
    fi
    echo "Connection failed, retrying in 3 seconds..."
    sleep 3
    retry_count=$((retry_count + 1))
done
