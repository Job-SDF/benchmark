import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class TimeSeriesAutoEncoder(nn.Module):
    def __init__(self, dim):
        super(TimeSeriesAutoEncoder, self).__init__()
        self.stable_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.trend_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.next_step_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        stable = self.stable_mlp(x)
        trend = self.trend_mlp(x)
        reconstructed = stable + trend  
        next_step = self.next_step_mlp(trend) + reconstructed  
        return reconstructed, next_step, stable, trend