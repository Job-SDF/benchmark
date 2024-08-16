set -o errexit
for dataset_name in 'r1' 'region' 'r2'
do
for model_name in 'PatchTST' 'FiLM' 'Koopa' 'DLinear' 'SegRNN'
do
for break_mode in 'bartlett' 'parzen' 'tukey-hanning' 'rayleigh'
do
CUDA_VISIBLE_DEVICES='1' python run.py --data job_demand_${dataset_name} --root_path ../../dataset/demand/ --data_path ${dataset_name}.parquet --model $model_name  --use_gpu 1 --task_mode '_break' --break_mode $break_mode
done
done
done