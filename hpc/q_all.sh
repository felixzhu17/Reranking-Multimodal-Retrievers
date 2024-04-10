#!/bin/bash

# Get the job IDs from squeue for user fz288
jobs=$(squeue -u fz288 -h -o %A)

# Iterate over each job ID and cancel it
for job in $jobs
do
  scancel $job
done