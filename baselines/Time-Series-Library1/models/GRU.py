import torch
import torch.nn as nn
from torch.autograd import Variable
class Model(nn.Module):

    def __init__(self, configs, num_layers=2, hidden_size=256):
        super(Model, self).__init__()
        self.input_size = configs.seq_len
        self.hidden_size =  hidden_size
        self.num_layers = num_layers
        self.pred_len = configs.pred_len
        
        self.gru = nn.GRU(input_size=configs.enc_in, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, configs.enc_in)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        
        # Propagate input through LSTM
        ula, hn = self.gru(x, h_0)
        
        out = self.fc(ula)[:,-self.pred_len:,:]
        
        return out