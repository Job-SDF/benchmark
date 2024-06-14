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
task1_rg_activate_fn='gelu'
task1_k=1
task1_num_epochs=1000
task1_lr=0.0001

lp_weight=0.0
rg_weight=0.001
rank_weight=0.0
diff_weight=0.0
adaptive='no'
task2_lr=0.001
task2_rg_loss_fn='mse'
task2_rg_activate_fn='gelu'
num_epochs=1000
k=5

task1_identity=epoch_${task1_num_epochs}_k_${task1_k}_lr_${task1_lr}_initalembed_yes_seed_$task1_seed/rglossfn_${task1_rg_loss_fn}_activate_${task1_rg_activate_fn}_rgweight_${task1_rg_weight}_lpweight_${task1_lp_weight}_rankweight_${task1_rank_weight}_conweight_${task1_con_weight}_gaussian_yes_crossattn_${task1_bias}_bias_${task1_bias}

for seed in 0
do

task2_identity=epoch_${num_epochs}_k_${k}_lr_${task2_lr}_initalembed_yes_seed_$seed/rglossfn_${task2_rg_loss_fn}_activate_${task1_rg_activate_fn}_rgweight_${rg_weight}_lpweight_${lp_weight}_rankweight_${rank_weight}_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes/diffweight_${diff_weight}_adaptive_${adaptive}

for ((month=$start_date; month<$end_date; month++ )) 
do

python code/train_task2.py \
    --data_name $data \
    --k $k \
    --mode train \
    --task task2 \
    --time yes \
    --num_epochs $num_epochs \
    --train_path data/$data/task2/$month/train/triplet_value.tsv \
    --eval_path data/$data/task2/$month/eval/triplet_value.tsv \
    --test_path data/$data/task2/$month/eval/triplet_value.tsv \
    --date $month \
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/${task1_identity}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/${task1_identity}_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn $task2_rg_loss_fn \
    --rg_activate_fn $task2_rg_activate_fn \
    --e_dim $e_dim \
    --adaptive $adaptive \
    --initial_embedding yes \
    --seed $seed \
    --lr $task2_lr

python code/test_task2.py \
    --data_name $data \
    --k $k \
    --mode test \
    --task task2 \
    --time yes \
    --num_epochs $num_epochs \
    --train_path data/$data/task2/$month/train/triplet_value.tsv \
    --eval_path data/$data/task2/$month/eval/triplet_value.tsv \
    --test_path data/$data/task2/$month/eval/triplet_value.tsv \
    --date $month \
    --load_state_path outputs/${owner_id}_checkpoints/task2/$data/${task1_identity}.pt \
    --load_node_embedding_path outputs/${owner_id}_node_embedding/task2/$data/train/${task1_identity}_node.pt \
    --load_time_embedding_path outputs/${owner_id}_time_embedding/task2/$data/train/$month/${task1_identity}/${task2_identity}_node.pt \
    --fix_model yes \
    --con_weight 0.0 \
    --rank_weight $rank_weight \
    --lp_weight $lp_weight \
    --rg_weight $rg_weight \
    --diff_weight $diff_weight \
    --rg_loss_fn $task2_rg_loss_fn \
    --rg_activate_fn $task2_rg_activate_fn \
    --e_dim $e_dim \
    --adaptive $adaptive \
    --initial_embedding yes  \
    --seed $seed \
    --lr $task2_lr
done
done
