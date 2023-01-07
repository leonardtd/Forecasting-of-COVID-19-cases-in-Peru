import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Dataset

def generate_sequences(df, n_past, n_future):
    X, y = list(), list()

    for i in range(n_past, len(df) - n_future +1):
        X.append(df[i - n_past:i, 0].reshape(-1,1))
        y.append(df[i:i + n_future, 0])

    return np.array(X), np.array(y)


class SequenceDataset(Dataset):
    def __init__(self, X, y):

        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        return self.X[i], self.y[i]
    
    
    
class LSTMRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers
        )
        
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        
        ### parameter init
        self._reset_parameters()
        
    def forward(self, x):
        
        out, (hn, cn) = self.rnn(x)
        out = hn.mean(dim=0) #mean accross LSTM layers: https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        
        out = self.hidden(out)
        out = F.gelu(out)
        out = F.dropout(out, p=0.2, training=self.training)
        
        out = self.out_lin(out)

        return out


    
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=1e-3)
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
          
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        
        
    def _reset_parameters(self):
        """
        Parameter initialization
        """
        for m in self.modules():
            if  isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif  isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
                for weight in m._all_weights:
                    if "weight" in weight:
                        nn.init.xavier_uniform_(getattr(rnn,weight))
                    if "bias" in weight:
                        nn.init.uniform_(getattr(rnn,weight))