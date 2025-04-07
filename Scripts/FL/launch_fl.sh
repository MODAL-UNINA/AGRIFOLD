#!/bin/bash
#SBATCH --job-name=federated_learning    
#SBATCH --output=output/%j_output.txt   
#SBATCH --error=output/%j_error.txt      
#SBATCH --nodes=4                       
#SBATCH --ntasks=13                     
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4               
#SBATCH --time=48:00:00                 
#SBATCH --mem=32GB                      
#SBATCH --partition=gpus                 

NTASKS_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | cut -d'(' -f1)
NUM_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
echo "Number of nodes: $NUM_NODES"
echo "Number of tasks per node: $NTASKS_PER_NODE"

srun \
    --output=output/%j_%2n_%2t_output.txt \
    --error=output/%j_%2n_%2t_error.txt \
    --ntasks-per-node=$NTASKS_PER_NODE \
    --nodes=$NUM_NODES \
    bash run_fl_inner.sh

