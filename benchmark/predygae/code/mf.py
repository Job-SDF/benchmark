import pandas as pd
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='month_r1')
parser.add_argument('--e_dim', type=int, default=10)
args = parser.parse_args()
triplet_percentage = pd.read_csv(f'data/{args.data_name}/task2/old_train_triplet_value.tsv', header=None, sep='\t')
entities = pd.read_csv(f'data/{args.data_name}/entities.dict', header=None, sep='\t')
map_dict = {v:i for i,v in zip(entities[0], entities[1])}
user_num = (entities[1]<30000).sum()
item_num = (entities[1]>=30000).sum()
triplet_percentage[0] = triplet_percentage[0].apply(lambda x: map_dict[x])
triplet_percentage[2] = triplet_percentage[2].apply(lambda x: map_dict[x])
matrix = torch.zeros(user_num, item_num)
matrix[triplet_percentage[0]][triplet_percentage[2] - user_num] = 1
import torch
import torch.optim as optim
n, m = matrix.shape
k = 10  

P = torch.rand(n, k, requires_grad=True)
Q = torch.rand(k, m, requires_grad=True)
optimizer = optim.Adam([P, Q], lr=0.01)
num_iterations = 10000
for i in range(num_iterations):
    optimizer.zero_grad()
    loss = ((torch.matmul(P, Q) - matrix) ** 2).mean()  
    loss.backward()  
    optimizer.step()  
    if i % 100 == 0:
        print(f"Iteration {i}: Loss {loss.item()}")
torch.save(torch.cat([P, Q.T], dim=0), f'data/{args.data_name}/task2/graph_initial_emb_dim_{args.e_dim}.pt')