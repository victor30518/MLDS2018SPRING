import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        super().__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, vid_feats):
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden
