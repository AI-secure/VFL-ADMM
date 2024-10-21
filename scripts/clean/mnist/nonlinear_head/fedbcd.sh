method='fedbcd'
dataset='mnist'
lr=0.1
dpsigma=20
clip=0.01
nround=30
bs=1024
seed=100
local_step=5

common_args="--vis --project vfl_${dataset} --method ${method}  --dataset ${dataset} --gpu_ids 0 --num_round ${nround} --out_dir iclr_output/${dataset}/nonlinear"

for server_model_type in 'nonliear2layer'  
do
scope_name="${method}_lr${lr}bs${bs}step${local_step}${server_model_type}"
task_params="${common_args} --learning_rate ${lr} --real_batch_size ${bs} --seed ${seed} --scope_name ${scope_name} --local_admm_epoch ${local_step} --server_model_type ${server_model_type}"
list_of_jobs="python run.py ${task_params}"
echo ${list_of_jobs}
${list_of_jobs}
done 

