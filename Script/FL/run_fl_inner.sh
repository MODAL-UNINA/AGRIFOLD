#!/bin/bash
#SBATCH --output=output/%j_%2n_%2t_output.txt    # Output del server
#SBATCH --error=output/%j_%2n_%2t_error.txt      # Errori del server

batch_size=256
epochs=5
num_rounds=100

echo "Start time: $(date)"

export NODENAMEFULL=$(hostname)
export NODEID=$(hostname | cut -d'.' -f1)
export NODEADDR="${NODENAMEFULL#"$NODEID"}"
export SERVERID=$( cut -d ',' -f1 <<< $SLURM_JOB_NODELIST | cut -d '-' -f1,2 | tr -d '[')
export SERVERNAMEFULL=$SERVERID$NODEADDR
export SERVERIP=$(getent hosts $SERVERID | awk '{ print $1 }')

# get slurm number of nodes
export NUM_TASKS=$SLURM_NTASKS
export NUM_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
export NUM_CLIENTS=$(($NUM_NODES-1))

# get current task id
export TASKID=$SLURM_PROCID
export GPUID=$SLURM_LOCALID
export OMP_NUM_THREADS=4


echo "SLURM node list: $SLURM_JOB_NODELIST"
echo "Server ID: $SERVERID"
echo "Server Name: $SERVERNAMEFULL"
echo "Server IP: $SERVERIP"
echo "Number of nodes: $NUM_NODES"
echo "Number of tasks: $NUM_TASKS"
echo "Node ID: $NODEID"
echo "Task ID: $TASKID"
echo "GPU ID: $GPUID"
echo "Node Name: $NODENAMEFULL"
echo "Number of gpus: $SLURM_STEP_GPUS | $SLURM_JOB_GPUS ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"

server_address="$SERVERIP:12389"

echo Creating runs folder
sleep $TASKID

mkdir -p runs/$SLURM_JOB_ID/$TASKID

source /lustre/home/fpiccialli/miniconda3/bin/activate environment

echo "Running nvidia-smi"

nvidia-smi

cp *.py *.json runs/$SLURM_JOB_ID/$TASKID


if [[ $TASKID -eq 0 ]]; then
    echo "Server"
    cd runs/$SLURM_JOB_ID
    ln -s ../../Preprocessed_dataset_new_conf_modified_without_enh ./
    echo "Preprocessed datasets folder linked"
    cd $TASKID
else
    echo "Client"
    cd runs/$SLURM_JOB_ID/$TASKID
    sleep 10
    while [ ! -d "../Preprocessed_dataset_new_conf_modified_without_enh" ]; do
        echo "Waiting for Preprocessed datasets folder to be created"
        sleep 10
    done
fi


if [[ $TASKID -eq 0 ]]; then
    PY_SCRIPT="server.py"

    PARAMS="--num-rounds=$num_rounds \
            --server-address=$server_address \
        "
    echo "Lancio server"

else
    PY_SCRIPT="client.py"

    client_id=$(($TASKID-1))

    
    PARAMS="    --gpuid=$GPUID \
                --batch-size=$batch_size \
                --epochs=$epochs \
                --server-address=$server_address \
                --client_id=$client_id \
                "

    echo "Lancio client $client_id con GPU $GPUID"
    sleep 5

fi



if [[ -f $PY_SCRIPT ]]; then
    echo "File trovato: $PY_SCRIPT"
    ls -l $PY_SCRIPT
else
    echo "Errore: Il file $PY_SCRIPT non esiste nella directory $(pwd)"
    exit 1
fi

echo "Directory corrente per TASK $TASKID: $(pwd)"
echo "PARAMS: $PARAMS"
echo "Running python script"
echo "Python Start time: $(date)"

python -uB $PY_SCRIPT $PARAMS


echo "End time: $(date)"

