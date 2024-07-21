#!/bin/bash
#SBATCH -J TEST_EVQA_Interaction_3_B_ckpt_model_step_3000
#SBATCH -A MLMI-fz288-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
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

# Set environment variables
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# SLURM provides these environment variables
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Set CUDA_VISIBLE_DEVICES to match the SLURM GPU allocation
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($SLURM_GPUS_ON_NODE - 1)))
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
python src/main.py --config configs/Rerank/evqa_experiments/evqa_interaction_B_3.jsonnet --mode test --experiment_name TEST_EVQA_Interaction_3_B_ckpt_model_step_3000_DDP_Test --tags "EVQA_Interaction_3_B" "test" --opts train.load_model_path="/home/fz288/rds/hpc-work/PreFLMR/experiments/EVQA_Interaction_3_B/train/saved_models/model_step_3000.ckpt" test.trainer_paras.limit_test_batches=5 > log_TEST_EVQA_Interaction_3_B_ckpt_model_step_3000 2>&1
