#!/bin/bash

# List of folders to move
folders=(
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_No_Vision_ckpt_model_step_3021"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_No_Vision"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_L_Freeze_Vision_ckpt_model_step_6042"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_L_Freeze_Vision_ckpt_model_step_3021"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_L_Freeze_Vision_ckpt_model_step_1007"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_L_Freeze_Vision"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_B_Neg_Sample_Train_on_Retrieve_ckpt_model_step_7552"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_B_Neg_Sample_ckpt_model_step_7552"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_B_Neg_Sample_ckpt_model_step_6042"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_B_Freeze_Vision_cross_encoder_num_hidden_layers3"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_B_Freeze_Vision_ckpt_model_step_3021"
    "/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_FLMRQuery_Full_Context_Rerank_B_Freeze_Vision"
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