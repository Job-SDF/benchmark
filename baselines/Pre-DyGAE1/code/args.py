import argparse
import os
import random
import numpy as np
import torch
import sys
import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default="month_r1", type=str)
parser.add_argument("--task", default="task2", type=str)
parser.add_argument("--rg_loss_fn", default="tweedie", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument(
    "--train_path", default="data/month_r1/task2/old_train_triplet_value.tsv", type=str)
parser.add_argument(
    "--test_path", default="data/month_r1/task2/old_test_triplet_value.tsv", type=str)
parser.add_argument(
    "--eval_path", default="data/month_r1/task2/old_eval_triplet_value.tsv", type=str)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--time", default="yes", type=str)
parser.add_argument("--fix_emb", default="no", type=str)
parser.add_argument("--fix_model", default="yes", type=str)
parser.add_argument("--cross_attn", default="yes", type=str)
parser.add_argument("--gaussian", default="yes", type=str)
parser.add_argument("--bias", default="yes", type=str)
parser.add_argument("--owner_id", default="new", type=str)
parser.add_argument("--rg_activate_fn", default="gelu", type=str)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--eval_step", default=100, type=int)
parser.add_argument("--e_dim", default=10, type=int)
parser.add_argument("--sample_size", default=3000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--rg_weight", default=100.0, type=float)
parser.add_argument("--lp_weight", default=1.0, type=float)
parser.add_argument("--rank_weight", default=0.0, type=float)
parser.add_argument("--con_weight", default=0.0, type=float)
parser.add_argument("--diff_weight", default=0.0, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--time_seed", default=-1, type=int)
parser.add_argument("--k", default=10, type=int)
parser.add_argument("--infer_interval", default=1, type=int)
parser.add_argument("--initial_embedding", default="yes", type=str)
parser.add_argument("--date", default=1, type=int)
parser.add_argument("--load_state_path", default='outputs/new_checkpoints/task2/month_r1/epoch_1000_k_1_lr_0.0001_initalembed_yes_seed_0/rglossfn_tweedie_activate_relu_rgweight_100.0_lpweight_1.0_rankweight_0.0_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes.pt', type=str)
parser.add_argument("--load_node_embedding_path", default='/individual/chenxi02/conferences/NeuIPS24/benchmark/baselines/Pre-DyGAE/outputs/new_node_embedding/task2/month_r1/train/epoch_1000_k_1_lr_0.0001_initalembed_yes_seed_0/rglossfn_tweedie_activate_relu_rgweight_100.0_lpweight_1.0_rankweight_0.0_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes_node.pt', type=str)
parser.add_argument("--load_time_embedding_path", default=None, type=str)
parser.add_argument("--start_date", default=24, type=int)
parser.add_argument("--end_date", default=30, type=int)
parser.add_argument("--task2_strategy", default="self", type=str)
parser.add_argument("--adaptive", default="no", type=str)
parser.add_argument("--task2_abalation", default="no", type=str)

parser.add_argument("--strategy", type=str, default='self')
parser.add_argument("--shift_type", type=str, default='time')
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--time_lr", type=float, default=0.0001)

parser.add_argument("--re_weight", type=float, default=1)
parser.add_argument("--ne_weight", type=float, default=1)
parser.add_argument("--com_weight", type=float, default=11)

parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--file_name", type=str,
                    default='epoch_1000_k_1_lr_0.0001_initalembed_yes_seed_0/rglossfn_tweedie_activate_relu_rgweight_100.0_lpweight_1.0_rankweight_0.0_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes/epoch_1000_k_3_lr_0.0001_initalembed_yes_seed_0/rglossfn_tweedie_activate_relu_rgweight_100.0_lpweight_1.0_rankweight_0.0_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes/diffweight_0.0_adaptive_no_node')
parser.add_argument("--save_file_name", type=str,
                    default='epoch_1000_k_1_lr_0.0001_initalembed_yes_seed_0/rglossfn_tweedie_activate_relu_rgweight_100_lpweight_1_rankweight_0_conweight_0_gaussian_yes_crossattn_yes_bias_yes/epoch_1000_k_3_lr_0.0001_initalembed_yes_seed_0/rglossfn_tweedie_activate_relu_rgweight_100.0_lpweight_1.0_rankweight_0.1_conweight_0.0_gaussian_yes_crossattn_yes_bias_yes/diffweight_1.0_adaptive_yes/lr_0.001_seed_0_epochs_1000_temperature_2.0_self_1_node')
args = parser.parse_args()


args.root_dir = f"epoch_{args.num_epochs}_k_{args.k}_lr_{args.lr}_initalembed_{args.initial_embedding}_seed_{args.seed}/"

args.identity = f"rglossfn_{args.rg_loss_fn}_activate_{args.rg_activate_fn}_rgweight_{args.rg_weight}_lpweight_{args.lp_weight}_rankweight_{args.rank_weight}_conweight_{args.con_weight}_gaussian_{args.gaussian}_crossattn_{args.cross_attn}_bias_{args.bias}"

if '0' in args.train_path:
    args.identity += "_" + args.train_path.split('/')[-2]

if args.time == "yes":
    assert args.task == 'task2'
    if args.task2_abalation == 'yes':
        args.identity += f"/diffweight_{args.diff_weight}_adaptive_{args.adaptive}_abalation"
    else:
        args.root_dir = args.load_state_path.split(
            '/')[-2] + '/' + args.load_state_path.split('/')[-1][:-3]
        args.identity = f"/epoch_{args.num_epochs}_k_{args.k}_lr_{args.lr}_initalembed_{args.initial_embedding}_seed_{args.seed}/rglossfn_{args.rg_loss_fn}_activate_{args.rg_activate_fn}_rgweight_{args.rg_weight}_lpweight_{args.lp_weight}_rankweight_{args.rank_weight}_conweight_{args.con_weight}_gaussian_{args.gaussian}_crossattn_{args.cross_attn}_bias_{args.bias}/diffweight_{args.diff_weight}_adaptive_{args.adaptive}"

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.checkpoint_path = f"outputs/{args.owner_id}_checkpoints/{args.task}/{args.data_name}/"
args.log_path = f"outputs/{args.owner_id}_logs/{args.task}/{args.data_name}/{args.mode}/"
args.results_path = f"outputs/{args.owner_id}_results/{args.task}/{args.data_name}/{args.mode}/"
args.scores_path = f"outputs/{args.owner_id}_scores/{args.task}/{args.data_name}/{args.mode}/"
args.node_embedding_path = f"outputs/{args.owner_id}_node_embedding/{args.task}/{args.data_name}/{args.mode}/"

if args.time == 'yes':
    args.log_path += f"{args.date}/"
    args.results_path += f"{args.date}/"
    args.scores_path += f"{args.date}/"
    args.checkpoint_path += f"{args.date}/"
    args.node_embedding_path += f"{args.date}/"

    args.old_triplet_path = f'data/{args.data_name}/task2/old_triplet_value.tsv'

    if args.mode == 'test':
        if len(args.load_time_embedding_path.split('/')) > 8:
            time_identify = args.load_time_embedding_path.split('/')[-1][:-3]
            args.identity += f"_{args.task2_strategy}_{time_identify}"

    args.time_embedding_path = f"outputs/{args.owner_id}_time_embedding/{args.task}/{args.data_name}/{args.mode}/{args.date}/"
    args.time_embedding_path += args.root_dir
    args.time_embedding_path += args.identity + "_node.pt"
    args.graph_inital_emb_path = f"data/{args.data_name}/task2/graph_initial_emb_dim_{args.e_dim}.pt"
    if not os.path.exists('/'.join(args.time_embedding_path.split('/')[:-1])):
        os.makedirs('/'.join(args.time_embedding_path.split('/')[:-1]))


args.checkpoint_path += args.root_dir
args.log_path += args.root_dir
args.results_path += args.root_dir
args.scores_path += args.root_dir
args.node_embedding_path += args.root_dir


args.checkpoint_path += args.identity + ".pt"

if args.time_seed != -1:
    assert args.time == 'yes' and args.task == 'task2'
    args.log_path += args.identity + f"_time_seed_{args.time_seed}.log"
else:
    args.log_path += args.identity + ".log"

args.results_path += args.identity + ".pt"
args.regression_path = args.scores_path + args.identity + '.npy'
args.scores_path += args.identity + ".pt"

args.node_embedding_path += args.identity + "_node.pt"


if not os.path.exists('/'.join(args.checkpoint_path.split('/')[:-1])):
    os.makedirs('/'.join(args.checkpoint_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.log_path.split('/')[:-1])):
    os.makedirs('/'.join(args.log_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.results_path.split('/')[:-1])):
    os.makedirs('/'.join(args.results_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.scores_path.split('/')[:-1])):
    os.makedirs('/'.join(args.scores_path.split('/')[:-1]))

if not os.path.exists('/'.join(args.node_embedding_path.split('/')[:-1])):
    os.makedirs('/'.join(args.node_embedding_path.split('/')[:-1]))



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(
        sys.stdout), logging.FileHandler(args.log_path)],
    level=logging.INFO
)

for arg_name, arg_value in vars(args).items():
    logger.info(f"{arg_name}: {arg_value}")
