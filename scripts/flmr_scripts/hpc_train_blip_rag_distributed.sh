#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Last updated: Fri 30 Jul 11:07:58 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J OKVQA_BLLP_TRAIN
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A BYRNE-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:4
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime.

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e 's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

if [ -z ${CONDA_ENV_PATH+x} ]; then
  echo "Please pass the absolute path to your conda environment by prepending CONDA_ENV_PATH=abs/path/to/the/args variable."
  exit
fi
#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh   # Leave this line (enables the module command)
module purge                  # Removes all modules still loaded
module load rhel8/default-amp # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load python/3.8
module load miniconda/3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PATH"
which python

if [ -z ${CHECKPOINT_DIRS+x} ]; then
  echo "Please pass the path to the directory of the checkpoint you want to run inference on by prepending
  CHECKPOINT_DIRS=abs/path/to/the/args to your command."
  exit
fi
if [ -z ${EXPERIMENT_NAMES+x} ]; then
  echo "Please specify the version of the data you want to run inference for by prepending VERSIONS=int to the command.
  For example, to run inference on version 1 pass prepend VERSIONS=1. This should match the /version_*/ portion of
  CHECKPOINT_DIRS"
  exit
fi


CHECKPOINT_DIRS=($CHECKPOINT_DIRS)
EXPERIMENT_NAMES=($EXPERIMENT_NAMES)
NUMBER_OF_MODELS="${#CHECKPOINT_DIRS[@]}"
NUMBER_OF_EXPERIMENT_NAMES="${#EXPERIMENT_NAMES[@]}"
if [ "$NUMBER_OF_MODELS" -ne "$NUMBER_OF_EXPERIMENT_NAMES" ]; then
  echo "Number of models is $NUMBER_OF_MODELS but only got $NUMBER_OF_EXPERIMENT_NAMES in the data versions array so cannot determine version. Aborting."
  exit
fi
if [ -z ${PROC_NUM_WORK+x} ]; then
  PROC_NUM_WORK=128
  echo "Preprocessing data on $PROC_NUM_WORK cores"
fi
if [ -z ${BATCH_SIZE+x} ]; then
  BATCH_SIZE=8
  echo "Decoding with batch size $BATCH_SIZE"
fi

CHECKPOINT_DIR="${CHECKPOINT_DIRS[$SLURM_ARRAY_TASK_ID]}"
EXPERIMENT_NAME="${EXPERIMENT_NAMES[$SLURM_ARRAY_TASK_ID]}"
#! Full path to application executable:
application="python src/main.py"

#! Run options for the application:
options="--experiment_name \"OKVQA_RAG_VisualColBERT_BLIP2_T5xl_with_pretrained_ViT(WIT)_10ROI_with_text_based_vision_K=5_LR1e-4_4GPUs\" \
    --config \"configs/rag/okvqa/RAG_colbert_BLIP2_with_vision.jsonnet\" \
    --accelerator auto --devices auto --strategy ddp \
    --reset --override \
    --num_sanity_val_steps 0 \
    --precision bf16 \
    --mode train \
    --opts train.trainer_paras.max_epochs=1000 \
             train.batch_size=2 \
             train.trainer_paras.val_check_interval=1000 \
             valid.batch_size=4 \
             train.trainer_paras.accumulate_grad_batches=16 \
             train.early_stopping_callback_paras.patience=5 \
             train.optimizer_config.optimizer_params.lr=0.0001 \
             train.optimizer_config.retriever_lr=0.0001 \
             train.optimizer_config.scheduler=none \
             train.model_checkpoint_callback_paras.save_top_k=1 \
             model_config.num_beams=2 \
             model_config.num_ROIs=9 \
             model_config.num_knowledge_passages=5"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR" # The value of SLURM_SUBMIT_DIR sets workdir to the directory
# in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$((${numnodes} * ${mpi_tasks_per_node}))

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"

if [ "$SLURM_JOB_NODELIST" ]; then
  #! Create a machine file:
  export NODEFILE=$(generate_pbs_nodefile)
  cat $NODEFILE | uniq >machine.file.$JOBID
  echo -e "\nNodes allocated:\n================"
  echo $(cat machine.file.$JOBID | sed -e 's/\..*$//g')
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD
