import argparse
import datetime
import random
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats.mstats import winsorize
from collections import deque
from copy import deepcopy
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--look_back_window', type=int, default=3)
parser.add_argument('--percentile', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.0008)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--min_delta', type=float, default=0.001)

# Model-specific Parameters
parser.add_argument('--d_model', type=int, default=120)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--dim_feedforward', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--depth', type=int, default=3)
args = parser.parse_args(args=[])

seed = args.seed
batch_size = args.batch_size
num_epochs = args.num_epochs
look_back_window = args.look_back_window
percentile = args.percentile
lr = args.lr
step_size = args.step_size
gamma = args.gamma
patience = args.patience
min_delta = args.min_delta

d_model = args.d_model
nhead = args.nhead
num_encoder_layers = args.num_encoder_layers
dim_feedforward = args.dim_feedforward
dropout = args.dropout
depth = args.depth

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = np.load('./data.npz')
col_ind = 0  # Select y[:,0]
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train'][:, col_ind]
y_val = data['y_val'][:, col_ind]
y_test = data['y_test'][:, col_ind]

# Look back


def look_back(x, window_size):
    temp = sliding_window_view(x, window_shape=(window_size, x.shape[1]), axis=(
        0, 1)).reshape(-1, window_size, x.shape[1]).copy()
    return Tensor(temp).reshape(-1, window_size * x.shape[1])


X_train_ts = look_back(X_train, look_back_window)
X_val_ts = look_back(X_val, look_back_window)

# Winsorization
y_train_ts = 100 * \
    Tensor(winsorize(y_train, [percentile, percentile])[
           (look_back_window-1):].reshape(-1, 1))
y_val_ts = 100 * \
    Tensor(winsorize(y_val, [percentile, percentile])
           [(look_back_window-1):].reshape(-1, 1))

# Dataset
train_set = DataLoader(CustomDataset(X_train_ts, y_train_ts),
                       batch_size=batch_size, shuffle=False)
val_set = DataLoader(CustomDataset(X_val_ts, y_val_ts),
                     batch_size=batch_size, shuffle=False)

model = Tranformer(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    look_back_window=look_back_window,
    depth=depth
)

model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=step_size, gamma=gamma)
time = datetime.datetime.now()
time = time.strftime('%Y-%m-%d-%H-%M-%S')
suffix = f'd_model{d_model}-nhead{nhead}-num_encoder_layers{num_encoder_layers}-dim_feedforward{dim_feedforward}-dropout{dropout}-depth{depth}-look_back_window{look_back_window}-lr{lr}-step_size{step_size}-gamma{gamma}'
writer = SummaryWriter(f'runs/Transformer/{time}_{suffix}')
ret_deque = deque(maxlen=patience)
para_deque = deque(maxlen=patience)
accu_deque = deque(maxlen=patience)

for epoch in range(1, num_epochs + 1):
    epoch_loss_train, cum_ret_train = train(
        iterator=train_set, optimizer=optimizer, criterion=criterion, device=device)
    epoch_loss_val, cum_ret_val, accuracy = evaluate(
        model=model, iterator=val_set, criterion=criterion, device=device)
    scheduler.step()
    ret_deque.append(cum_ret_val)
    accu_deque.append(accuracy)
    writer.add_scalar('Accumulated Return / validation', cum_ret_val, epoch)
    writer.add_scalar('Accuracy / validation', accuracy, epoch)
    writer.add_scalar('l2 loss / train', epoch_loss_train, epoch)
    writer.add_scalar('l2 loss / validation', epoch_loss_val, epoch)
    para_deque.append(model.state_dict())
    print(
        f'Cumret in validation: {cum_ret_val: .4f}       Accuracy: {accuracy: .4f}')
    if (len(ret_deque) >= patience and max(ret_deque) == ret_deque[0]) or epoch == num_epochs:
        model_index = np.array(accu_deque).argmax()
        torch.save(para_deque[model_index],
                   f'./checkpoints/Transformer/{time}_{suffix}')
        del para_deque
        break
