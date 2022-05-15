import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import random
import datetime
import os
import copy
from collections import deque
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--look_back_window', type=int, default=3)
parser.add_argument('--percentile', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.0032)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.901)
parser.add_argument('--patience', type=int, default=10)

# Model-specific Parameters
parser.add_argument('--input_size', type=int, default=120)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.07)
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

input_size = args.input_size
hidden_size = args.hidden_size
num_layers = args.num_layers
dropout = args.dropout

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
source_path = Path('./data.npz')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ret_deque = deque(maxlen=patience)
para_deque = deque(maxlen=patience)
accu_deque = deque(maxlen=patience)
filename = f'input_size{input_size}-hidden_size{hidden_size}-num_layers{num_layers}-dropout{dropout}-look_back_window{look_back_window}-lr{lr}-step_size{step_size}-gamma{gamma}-patience{patience}'
time = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
Path('./log/LSTM').mkdir(parents=True, exist_ok=True)
Path('./log/LSTM/checkpoints').mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(f'./log/LSTM/runs/{time}/{filename}')

LSTM = BaselineLSTM(batch_size, look_back_window, percentile, lr, step_size,
                    gamma, input_size, hidden_size, num_layers, dropout, source_path, device)
LSTM.pack_data_('./temp.npz')

for epoch in range(1, num_epochs + 1):
    epoch_loss_train, cum_ret_train = LSTM.train()
    epoch_loss_val, cum_ret_val, accuracy = LSTM.evaluate()
    LSTM.scheduler.step()
    ret_deque.append(cum_ret_val)
    accu_deque.append(accuracy)
    para_deque.append(LSTM.model.state_dict())
    print(f'Epoch: {epoch}\tCumret_val: {cum_ret_val: .4f}\tAccuracy: {accuracy: .4f}')
    writer.add_scalar('Accumulated Return / validation', cum_ret_val, epoch)
    writer.add_scalar('Accuracy / validation', accuracy, epoch)
    writer.add_scalar('l2 loss / train', epoch_loss_train, epoch)
    writer.add_scalar('l2 loss / validation', epoch_loss_val, epoch)
    if (len(ret_deque) >= patience and max(ret_deque) == ret_deque[0]) or epoch == num_epochs:
        model_index = np.array(accu_deque).argmax()
        torch.save(para_deque[model_index], f'./log/LSTM/checkpoints/{time}_{filename}')
        del para_deque
        break