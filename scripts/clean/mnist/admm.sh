method='admm'
dataset='mnist'
lr=0.1
admmep=20
rho=1
dpsigma=20
clip=0.01
nround=30
bs=1024
seed=100

common_args=" --project vfl_${dataset} --method ${method}  --dataset ${dataset} --gpu_ids 0 --num_round ${nround} --out_dir iclr_output/${dataset}/clean"


scope_name="${method}_lr${lr}bs${bs}admmep${admmep}rho${rho}"
task_params="${common_args} --save_model --learning_rate ${lr} --real_batch_size ${bs} --seed ${seed} --scope_name ${scope_name}"
admm_args="--local_admm_epoch ${admmep} --rho ${rho}"
load_params="--eval_only --load_model_dir iclr_output/${dataset}/clean/${scope_name}/model${seed}"
list_of_jobs="python run.py ${task_params} ${admm_args} ${load_params}"
echo ${list_of_jobs}
${list_of_jobs}


