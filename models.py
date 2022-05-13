import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import datetime
import os
import copy
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats.mstats import winsorize


def look_back(x, window_size):
    temp = sliding_window_view(x, window_shape=(window_size, x.shape[1]), axis=(
        0, 1)).reshape(-1, window_size, x.shape[1]).copy()
    return Tensor(temp).reshape(-1, window_size * x.shape[1])


def load_data(path):
    data = np.load(path)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    return X_train, X_val, X_test, y_train, y_val, y_test


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, look_back_window, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout)
        self.activation = nn.Tanh
        self.linear1 = nn.Linear(
            hidden_size * look_back_window, hidden_size * look_back_window)
        self.linear2 = nn.Linear(hidden_size * look_back_window, 1)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2]))
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, look_back_window, depth, batch_first=True, norm_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first, norm_first)
        self.depth = depth

    def forward(self, x):
        for i in range(self.depth):
            x = self.encoder(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, look_back_window, batch_first=True):
        super(TransformerDecoderLayer, self).__init__()
        self.linear1 = nn.Linear(
            d_model * look_back_window, dim_feedforward * look_back_window)
        self.linear2 = nn.Linear(dim_feedforward * look_back_window, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, look_back_window, depth, num_decoder_layers=1, batch_first=True, norm_first=True):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first, norm_first, look_back_window, depth)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(
            d_model, dim_feedforward, dropout, batch_first, look_back_window)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, x):
        mask = np.tril(np.ones(x.shape[0]**2).reshape(x.shape[0], -1))
        x = self.encoder(x, mask)
        x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2]))
        x = self.decoder(x)
        return x


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.data = torch.cat([x, y], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx, :-1]
        label = self.data[idx, -1:]
        sample = {'feature': feature, 'label': label}
        return sample


class Base:
    def __init__(self, batch_size, look_back_window, percentile, source_path, device):
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.source_path = source_path
        self.device = device

        self.criterion = nn.MSELoss()
        self.pack_data(percentile, source_path)

    def pack_data(self, percentile, source_path):
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(source_path)
        X_train_ts = look_back(X_train, self.look_back_window)
        X_val_ts = look_back(X_val, self.look_back_window)
        y_train_ts = 100 * Tensor(winsorize(y_train, [percentile, percentile])[
                                  (self.look_back_window-1):].reshape(-1, 1))
        y_val_ts = 100 * Tensor(winsorize(y_val, [percentile, percentile])[
                                (self.look_back_window-1):].reshape(-1, 1))
        self.train_set = DataLoader(CustomDataset(
            X_train_ts, y_train_ts), batch_size=self.batch_size, shuffle=False)
        self.val_set = DataLoader(CustomDataset(
            X_val_ts, y_val_ts), batch_size=self.batch_size, shuffle=False)

    def train(self):
        self.model.train()
        iterator = self.train_set
        epoch_loss = 0
        cum_ret = 0
        for i, batch in enumerate(iterator):
            feature = batch['feature'].reshape(
                batch['feature'].shape[0], self.look_back_window, self.num_features).to(self.device)
            label = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(feature).to(self.device)
            batch_loss = self.criterion(output, label)
            batch_loss.backward()
            self.optimizer.step()
            output = (output - output.mean()) / output.std()
            cum_ret += (output * label).sum()
            epoch_loss += batch_loss
        return (epoch_loss / len(iterator)).item(), cum_ret.item()

    def evaluate(self):
        self.model.eval()
        iterator = self.val_set
        epoch_loss = 0
        cum_ret = 0
        num_correct = 0
        num_wrong = 0
        with torch.no_grad():
            for _, batch in enumerate(iterator):
                feature = batch['feature'].reshape(
                    batch['feature'].shape[0], self.look_back_window, self.num_features).to(self.device)
                label = batch['label'].to(self.device)
                output = self.model(feature)
                batch_loss = self.criterion(output, label)
                output = (output - output.mean()) / output.std()
                cum_ret += (output * label).sum()
                epoch_loss += batch_loss
                num_correct += (output * label > 0).sum()
                num_wrong += (output * label < 0).sum()
        return (epoch_loss / len(iterator)).item(), cum_ret.item(), (num_correct / (num_correct + num_wrong)).item()


class BaselineLSTM(Base):
    def __init__(self, batch_size, look_back_window, percentile, lr, step_size, gamma, input_size, hidden_size, num_layers, dropout, source_path, device):
        super().__init__(batch_size, look_back_window, percentile, source_path, device)
        self.num_features = input_size
        self.save_dir = Path('BaselineLSTM')
        self.model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            look_back_window=look_back_window
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)


class AlphaTransformer(Base):
    def __init__(self, batch_size, look_back_window, percentile, lr, step_size, gamma, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, depth, source_path, device):
        super().__init__(batch_size, look_back_window, percentile, source_path, device)
        self.num_features = d_model
        self.save_dir = Path('AlphaTransformer')
        self.model = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            look_back_window=look_back_window,
            depth=depth
        ).to(self.device)
        self.optimizer = self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)
