import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, look_back_window, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first, dropout)
        self.activation = nn.Tanh
        self.linear1 = nn.Linear(hidden_size * look_back_window, hidden_size * look_back_window)
        self.linear2 = nn.Linear(hidden_size * look_back_window, 1)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = self.reshape(x.shape[0], (x.shape[1] * x.shape[2]))
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, look_back_window, depth, batch_first=True, norm_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first, norm_first)
        self.depth = depth

    def forward(self, x):
        for i in range(self.depth):
            x = self.encoder(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, look_back_window, batch_first=True):
        super(TransformerDecoderLayer, self).__init__()
        self.linear1 = nn.Linear(d_model * look_back_window, dim_feedforward * look_back_window)
        self.linear2 = nn.Linear(dim_feedforward * look_back_window, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, custom_encoder, custom_decoder, look_back_window, depth, num_decoder_layers=1, batch_first=True, norm_first=True):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first, norm_first, look_back_window, depth)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward, dropout, batch_first, look_back_window)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, x):
        ## MASK
        mask = 1
        x = self.encoder(x, mask)
        x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2]))
        x = self.decoder(x)
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.data = torch.cat([x,y], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx,:-1]
        label = self.data[idx,-1:]
        sample = {'feature': feature, 'label': label}
        return sample

def train(iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    cum_ret = 0
    for i, batch in enumerate(iterator):
        feature = batch['feature'].rehsape()
        
    pass

def evaluate(model, iterator, criterion, device):
    model.eval()     
    epoch_loss = 0    
    cum_ret = 0
    num_correct = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
    pass



