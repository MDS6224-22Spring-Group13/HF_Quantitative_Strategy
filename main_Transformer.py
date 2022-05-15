import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import random
import datetime
from collections import deque
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--look_back_window', type=int, default=3)
parser.add_argument('--percentile', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.0006)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.953)
parser.add_argument('--patience', type=int, default=20)

# Model-specific Parameters
parser.add_argument('--d_model', type=int, default=120)
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--dim_feedforward', type=int, default=256)
parser.add_argument('--dropout_e', type=float, default=0.52)
parser.add_argument('--dropout_d', type=float, default=0.09)
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

d_model = args.d_model
nhead = args.nhead
num_encoder_layers = args.num_encoder_layers
dim_feedforward = args.dim_feedforward
dropout_e = args.dropout_e
dropout_d = args.dropout_d
depth = args.depth

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
filename = f'd_model{d_model}-nhead{nhead}-num_encoder_layers{num_encoder_layers}-dim_feedforward{dim_feedforward}-dropout_e{dropout_e}-dropout_d{dropout_d}-depth{depth}-look_back_window{look_back_window}-lr{lr}-step_size{step_size}-gamma{gamma}-patience{patience}'
time = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
Path('./log/Transformer').mkdir(parents=True, exist_ok=True)
Path('./log/Transformer/checkpoints').mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(f'./log/Transformer/runs/{time}/{filename}')

Transformer = AlphaTransformer(batch_size, look_back_window, percentile, lr, step_size, gamma, d_model,
                               nhead, num_encoder_layers, dim_feedforward, dropout_e, dropout_d, depth, source_path, device)

for epoch in range(1, num_epochs + 1):
    epoch_loss_train, cum_ret_train = Transformer.train()
    epoch_loss_val, cum_ret_val, accuracy = Transformer.evaluate()
    Transformer.scheduler.step()
    ret_deque.append(cum_ret_val)
    accu_deque.append(accuracy)
    para_deque.append(Transformer.model.state_dict())
    print(
        f'Epoch: {epoch}\tCumret_val: {cum_ret_val: .4f}\tAccuracy: {accuracy: .4f}')
    writer.add_scalar('Accumulated Return / validation', cum_ret_val, epoch)
    writer.add_scalar('Accuracy / validation', accuracy, epoch)
    writer.add_scalar('l2 loss / train', epoch_loss_train, epoch)
    writer.add_scalar('l2 loss / validation', epoch_loss_val, epoch)
    if (len(ret_deque) >= patience and max(ret_deque) == ret_deque[0]) or epoch == num_epochs:
        model_index = np.array(accu_deque).argmax()
        torch.save(para_deque[model_index],
                   f'./log/Transformer/checkpoints/{time}_{filename}')
        del para_deque
        break
