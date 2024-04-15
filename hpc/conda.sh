module purge
module load slurm
# module load rhel8/default-amp
module load anaconda/3.2019-10
conda activate /home/fz288/rds/hpc-work/PreFLMR/VQA
export PATH=/home/fz288/rds/hpc-work/PreFLMR/VQA/bin:$PATH
module load cuda/11.8
export CUDA_HOME=$CUDA_INSTALL_PATH
# module load gcc/9.4.0