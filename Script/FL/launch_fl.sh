#!/bin/bash
#SBATCH --job-name=federated_learning    # Nome del job
#SBATCH --output=output/%j_output.txt    # Output del server
#SBATCH --error=output/%j_error.txt      # Errori del server
#SBATCH --nodes=4                       # Numero di nodi (uno per il server)
#SBATCH --ntasks=13                     # Numero di task (uno per il server)
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4               # Numero di CPU per task
#SBATCH --time=48:00:00                 # Tempo massimo di esecuzione
#SBATCH --mem=32GB                      # Memoria richiesta
#SBATCH --partition=gpus                 # Partizione GPU (modificare se necessario)

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

