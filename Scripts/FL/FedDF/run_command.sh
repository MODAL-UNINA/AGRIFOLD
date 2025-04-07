$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch veca --complex_arch master=veca,worker=veca --experiment demo \
    --data plants --pin_memory True --batch_size 64 --num_workers 6 \
    --partition_data non_iid_dirichlet --non_iid_alpha 0 --train_data_ratio 1 --val_data_ratio 0 \
    --n_clients 12 --participation_ratio 1 --n_comm_rounds 100 --local_n_epochs 5 --world 5,0,0,1,1,2,2,3,3,4,4,5,5 --on_cuda True \
    --fl_aggregate scheme=federated_average,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=plants,data_percentage=1.0,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
    --optimizer sgd --lr 0.0001 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 42 --pn_normalize True --same_seed_process False