set -o errexit
for dataset_name in 'r1' 'region' 'r2'
do
for model_name in 'PatchTST' 'FiLM' 'Koopa' 'DLinear' 'SegRNN'
do
python run.py --data job_demand_${dataset_name} --root_path ../../dataset/demand/ --data_path ${dataset_name}.parquet --model $model_name  --use_gpu 1 --task_mode ''
done
done