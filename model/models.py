import torch
import torch.nn as nn
import torch.nn.functional as F

class gfNN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, droprate):
        super(gfNN, self).__init__()
        self.graph_aug_linear = nn.Linear(in_features=n_feat, out_features=n_hid, bias=enable_bias)
        self.linear = nn.Linear(in_features=n_hid, out_features=n_class, bias=enable_bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.graph_aug_linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.log_softmax(x)

        return x