#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
  echo "Usage: source train_job.sh <script_name>"
  return 1
fi

SCRIPT_NAME=$1
APPLICATION="job_scripts/$SCRIPT_NAME.sh"

# Check if the script exists
if [ ! -f "$APPLICATION" ]; then
  echo "Error: $APPLICATION does not exist."
  return 1
fi

# Export the environment variable and submit the job
sbatch --export=SCRIPT_NAME="$SCRIPT_NAME" train_job
