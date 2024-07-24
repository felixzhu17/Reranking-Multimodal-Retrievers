#!/bin/bash
#SBATCH -J TEST_OKVQA_Interaction_MORES_5_B_Vanilla_Neg_Sample_Train_on_Retrieve_ckpt_model_step_2014
#SBATCH -A MLMI-fz288-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

. /etc/profile.d/modules.sh
source /home/${USER}/.bashrc

workdir="/home/fz288/rds/hpc-work/PreFLMR"
cd $workdir
echo -e "Changed directory to `pwd`.
"

set --
conda deactivate
conda deactivate
source hpc/conda.sh

export OMP_NUM_THREADS=1

num_gpus=$SLURM_GPUS_ON_NODE
gpu_indices=$(seq -s, 0 $(($num_gpus - 1)))
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID
======"
echo "Time: `date`"
echo -e "Timestamp: $TIMESTAMP
======"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
python src/main.py --config configs/Rerank/interaction_experiments/okvqa_interaction_B_mores_5_vanilla_neg_sample_retrieved.jsonnet --mode test --experiment_name TEST_OKVQA_Interaction_MORES_5_B_Vanilla_Neg_Sample_Train_on_Retrieve_ckpt_model_step_2014 --tags "OKVQA_Interaction_MORES_5_B_Vanilla_Neg_Sample_Train_on_Retriev" "test" --opts train.load_model_path="/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_Interaction_MORES_5_B_Vanilla_Neg_Sample_Train_on_Retrieve_ckpt_model_step_11077/train/saved_models/model_step_2014.ckpt" > log_TEST_OKVQA_Interaction_MORES_5_B_Vanilla_Neg_Sample_Train_on_Retrieve_ckpt_model_step_2014 2>&1
