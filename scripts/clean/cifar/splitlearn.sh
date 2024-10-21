method='vimsgd_concat'
dataset='cifar'
lr=0.008
dpsigma=20
clip=0.01
nround=50
bs=1024
seed=100

common_args="--vis --project vfl_${dataset} --method ${method} --drop_th 10 --dataset ${dataset} --gpu_ids 0 --num_round ${nround} --out_dir iclr_output/${dataset}/clean --features_split 0 1 2 3 4 5 6 7 8 9"


scope_name="${method}_lr${lr}bs${bs}"
task_params="${common_args} --learning_rate ${lr} --real_batch_size ${bs} --seed ${seed} --scope_name ${scope_name}"
list_of_jobs="python run.py ${task_params}"
echo ${list_of_jobs}
${list_of_jobs}

