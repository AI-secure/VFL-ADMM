method='admm'
dataset='modelnet40'
lr=0.1  
admmep=5
rho=0.1  
dpsigma=20
clip=0.01
nround=10
bs=1024
seed=100



common_args="--vis --project vfl_${dataset} --method ${method}  --drop_th 50  --dataset ${dataset} --gpu_ids 0 --num_round ${nround} --out_dir iclr_output/${dataset}/nonlinear --features_split 0 1 2 3 4"

for server_model_type in 'nonliear2layer' 
do
scope_name="${method}_lr${lr}bs${bs}admmep${admmep}rho${rho}${server_model_type}"
task_params="${common_args} --learning_rate ${lr} --real_batch_size ${bs} --seed ${seed} --scope_name ${scope_name} --server_model_type ${server_model_type}"
admm_args="--local_admm_epoch ${admmep} --rho ${rho}"
list_of_jobs="python run.py ${task_params} ${admm_args}"
echo ${list_of_jobs}
${list_of_jobs}
done
