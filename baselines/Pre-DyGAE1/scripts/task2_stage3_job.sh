set -o errexit

data=$1
start_date=$2
end_date=$3
task=task2
e_dim=10
owner_id=new

task1_seed=0
task1_rg_weight=100.0
task1_lp_weight=1.0
task1_rank_weight=0.0
task1_con_weight=0.0
task1_rg_loss_fn='tweedie'
task1_bias=yes
task1_k=1
task1_num_epochs=1000
task1_lr=0.0001

lp_weight=1.0
rg_weight=100.0
rank_weight=0.0
diff_weight=0.0
adaptive='no'
task2_lr=0.0001
num_epochs=1000
k=3


task1_identity=epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_relu_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}

for seed in 0
do

task2_identity=epoch_${num_epochs}_k_${k}_lr_${task2_lr}_initalembed_yes_seed_$seed/rglossfn_tweedie_activate_relu_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_${rank_weight}_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes/diffweight_${diff_weight}_adaptive_${adaptive}



epochs=1000
time_lr=0.001
temperature=2.0
for infer_interval in 1
do
for time_seed in 0
do
for strategy in 'self' 'mean' 'next'
do

python code/temporal_shift_infer.py \
    --infer_interval $infer_interval \
    --data_name $data \
    --owner_id $owner_id \
    --epochs $epochs \
    --start_date $start_date \
    --end_date $end_date \
    --lr $time_lr \
    --temperature $temperature \
    --file_name ${task1_identity}/${task2_identity}_node \
    --save_file_name ${task1_identity}/${task2_identity}/lr_${time_lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature}_${strategy}_${infer_interval}_node \
    --strategy $strategy \
    --time_seed $time_seed \
    --k $k \
    --mode train \
    --task task2 \
    --time yes \
    --num_epochs $num_epochs \
    --train_path data/$data/task2/old_train_triplet_value.tsv \
    --eval_path data/$data/task2/old_eval_triplet_value.tsv \
    --test_path data/$data/task2/old_test_triplet_value.tsv \
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/${task1_identity}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/${task1_identity}_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn relu \
    --e_dim $e_dim \
    --time_seed $time_seed \
    --task2_strategy $strategy \
    --adaptive $adaptive \
    --initial_embedding yes  \
    --seed $seed \
    --lr $task2_lr

python code/test_task2.py \
    --data_name $data \
    --k $k \
    --mode test \
    --task task2 \
    --time yes \
    --num_epochs $num_epochs \
    --train_path data/$data/task2/$end_date/train/triplet_value.tsv \
    --eval_path data/$data/task2/$end_date/triplet_value.tsv \
    --test_path data/$data/task2/$end_date/triplet_value.tsv \
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/${task1_identity}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/${task1_identity}_node.pt \
    --load_time_embedding_path outputs/${owner_id}_time_embedding/task2/$data/test/$end_date/${task1_identity}/${task2_identity}/lr_${time_lr}_seed_${time_seed}_epochs_${epochs}_temperature_${temperature}_${strategy}_${infer_interval}_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn tweedie \
    --rg_activate_fn relu \
    --e_dim $e_dim \
    --time_seed $time_seed \
    --task2_strategy $strategy \
    --adaptive $adaptive \
    --initial_embedding yes  \
    --seed $seed \
    --lr $task2_lr
    # --date $vali_end_date \
done
done
done
done

