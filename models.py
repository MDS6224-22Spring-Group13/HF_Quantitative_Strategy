import torch
import torch.nn as nn
import torch.nn.functional as F
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
        0, 1)).reshape(-1, window_size, x.shape[1])[:, ::-1, :].copy()
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
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size * look_back_window, 1)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2]))
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, depth, batch_first=True, norm_first=True, layer_norm_eps: float = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.depth = depth

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for i in range(self.depth):
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x),
                                       src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(
                    x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, look_back_window, batch_first=True):
        super(TransformerDecoder, self).__init__()
        self.linear1 = nn.Linear(
            d_model * look_back_window, dim_feedforward * look_back_window)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward * look_back_window, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout_e, dropout_d, look_back_window, depth, batch_first=True, norm_first=True):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout_e, depth)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers)
        self.transformer_decoder = TransformerDecoder(
            d_model, dim_feedforward, dropout_d, look_back_window)

    def forward(self, x, src_mask=None):
        x = self.transformer_encoder(x, src_mask)
        x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2]))
        x = self.transformer_decoder(x)
        return x


def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


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
    def __init__(self, batch_size, look_back_window, percentile, lr, step_size, gamma, d_model, nhead, num_encoder_layers, dim_feedforward, dropout_e, dropout_d, depth, source_path, device):
        super().__init__(batch_size, look_back_window, percentile, source_path, device)
        self.num_features = d_model
        self.src_mask = generate_square_subsequent_mask(
            self.look_back_window).to(self.device)
        self.model = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_e=dropout_e,
            dropout_d=dropout_d,
            look_back_window=look_back_window,
            depth=depth
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

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
            output = self.model(feature, self.src_mask).to(self.device)
            batch_loss = self.criterion(output, label)
            batch_loss.backward()
            self.optimizer.step()
            output = (output - output.mean()) / output.std()
            cum_ret += (output * label).sum()
            epoch_loss += batch_loss
        return (epoch_loss / len(iterator)).item(), cum_ret.item()
