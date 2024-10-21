method='vimsgd_concat'
dataset='nus'
lr=0.3
dpsigma=20
clip=0.01
nround=15
bs=1024
seed=100

common_args="--vis --project vfl_${dataset} --method ${method}  --dataset ${dataset} --gpu_ids 0 --num_round ${nround} --out_dir iclr_output/${dataset}/nonlinear --features_split 0 300 634 1134 1634"

for lr in  0.2 0.3 0.1
do
for server_model_type in  'nonliear3layer' 
do
scope_name="${method}_lr${lr}bs${bs}${server_model_type}"
task_params="${common_args} --learning_rate ${lr} --real_batch_size ${bs} --seed ${seed} --scope_name ${scope_name} --server_model_type ${server_model_type}"
list_of_jobs="python run.py ${task_params}"
echo ${list_of_jobs}
${list_of_jobs}
done 
done


