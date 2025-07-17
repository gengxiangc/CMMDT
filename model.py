

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import math

import torch.nn.functional as F
class CMMTransformer(nn.Module):
    def __init__(self,
                 in_channels=5,
                 conv_channels=(32, 32, 32),
                 dropout_rate=0.01,
                 out_dim=1,
                 task_type='regression',  # or 'classification'
                 d_model = 32
                 ):
        super(CMMTransformer, self).__init__()
        self.task_type = task_type.lower()
        assert self.task_type in ['regression',
                                  'classification'], "task_type must be 'regression' or 'classification'"

        self.conv1 = nn.Conv1d(in_channels, conv_channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(conv_channels[2])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels[2],
            nhead=4,
            dim_feedforward=32,
            dropout=dropout_rate,
            batch_first=False
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self._init_fc_input_size(in_channels)

        self.fc1 = nn.Linear(self.fc_input_size, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(16, out_dim)  

    def _init_fc_input_size(self, in_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 1000)
            x = self.forward_conv(dummy_input)
            pe = self.positional_encoding(x.size(2), x.size(1)).to(x.device)
            pe = pe.permute(0, 2, 1)
            x = x + pe
            x = x.permute(2, 0, 1)
            x = self.encoder(x)
            x = x.permute(1, 2, 0)
            self.fc_input_size = x.shape[1] * x.shape[2]

    def positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def forward_conv(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_conv(x)

        pe = self.positional_encoding(x.size(2), x.size(1)).to(x.device)
        pe = pe.permute(0, 2, 1)
        x = x + pe

        x = x.permute(2, 0, 1)
        x = self.encoder(x)
        x = x.permute(1, 2, 0)

        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
