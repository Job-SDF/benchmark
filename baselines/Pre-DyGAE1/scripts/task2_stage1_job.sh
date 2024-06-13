set -o errexit

data=$1
task=task2
e_dim=10
owner_id=new

task1_seed=0
task1_rg_weight=100
task1_lp_weight=1
task1_rank_weight=0
task1_con_weight=0
task1_rg_loss_fn='tweedie'
task1_bias=yes
task1_k=1
task1_num_epochs=1000
task1_lr=0.0001
rg_activate_fn='gelu'

python code/train_task1.py \
    --mode train \
    --data_name $data \
    --rg_weight $task1_rg_weight \
    --lp_weight $task1_lp_weight \
    --rg_loss_fn $task1_rg_loss_fn \
    --train_path data/$data/$task/old_train_triplet_value.tsv \
    --eval_path data/$data/$task/old_eval_triplet_value.tsv \
    --test_path data/$data/$task/old_train_triplet_value.tsv \
    --task $task \
    --k $task1_k \
    --num_epochs $task1_num_epochs \
    --owner_id $owner_id \
    --rg_activate_fn $rg_activate_fn \
    --rank_weight $task1_rank_weight \
    --gaussian yes \
    --bias $task1_bias \
    --cross_attn $task1_bias \
    --con_weight $task1_con_weight \
    --initial_embedding yes \
    --seed $task1_seed \
    --lr $task1_lr \
    --time no

python code/mf.py --data_name $data --e_dim $e_dim
