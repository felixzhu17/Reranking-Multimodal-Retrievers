#!/bin/bash
#SBATCH -J OKVQA_Decoder_Only_Rerank_ckpt_model_step_2002
#SBATCH -A MLMI-fz288-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
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

python src/main.py --config configs/Rerank/decoder_experiments/decoder_only_rerank.jsonnet --mode train --experiment_name OKVQA_Decoder_Only_Rerank_ckpt_model_step_2002 --reset --override --tags "OKVQA_Decoder_Only_Rerank" "train" --opts train.load_model_path="/home/fz288/rds/hpc-work/PreFLMR/experiments/OKVQA_Decoder_Only_Rerank/train/saved_models/model_step_2002.ckpt" > log_OKVQA_Decoder_Only_Rerank_ckpt_model_step_2002 2>&1
