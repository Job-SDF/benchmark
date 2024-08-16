set -o errexit
model_name=$1
for dataset_name in 'region' 'r1' 'r2'
do
python run.py --data job_demand_${dataset_name} --root_path ../../dataset/demand/ --data_path ${dataset_name}.parquet --model $model_name --use_gpu 0 --task_mode '_low'
done