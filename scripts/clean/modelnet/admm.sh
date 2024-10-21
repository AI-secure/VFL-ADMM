method='admm'
dataset='modelnet40'
lr=0.05
admmep=5
rho=0.5
dpsigma=20
clip=0.01
nround=10
bs=1024
seed=100



common_args="--vis --project vfl_${dataset} --method ${method}  --dataset ${dataset} --gpu_ids 0 --num_round ${nround} --out_dir iclr_output/${dataset}/clean --features_split 0 1 2 3 4"

for rho in 0.1 
do
for lr in 0.1
do
for admmep in 1
do
scope_name="${method}_lr${lr}bs${bs}admmep${admmep}rho${rho}"
task_params="${common_args} --learning_rate ${lr} --real_batch_size ${bs} --seed ${seed} --scope_name ${scope_name}"
admm_args="--local_admm_epoch ${admmep} --rho ${rho}"
list_of_jobs="python run.py ${task_params} ${admm_args}"
echo ${list_of_jobs}
${list_of_jobs}
done
done 
done 


