#!/bin/bash

# List of folders to move
folders=(
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/EVQA_Encoder_Decoder_Head_Rerank_Vanilla_Neg_Sample_Retrieved_ckpt_model_step_2000"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/EVQA_Encoder_Decoder_Head_Rerank_Neg_Sample_Retrieved_ckpt_model_step_2000"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_Decoder_Head_Rerank_Vanilla_Neg_Sample_Retrieved_ckpt_model_step_2002"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_Decoder_Head_Rerank_Neg_Sample_Retrieved_ckpt_model_step_2002"
)

# Target directory
target_dir="/home/fz288/rds/rds-cvnlp-hirYTW1FQIw/shared_space/fz288/experiments"

# Loop through each folder and move it to the target directory
for folder in "${folders[@]}"; do
    mv "$folder" "$target_dir"
    if [[ $? -eq 0 ]]; then
        echo "Moved $folder to $target_dir"
    else
        echo "Failed to move $folder"
    fi
done