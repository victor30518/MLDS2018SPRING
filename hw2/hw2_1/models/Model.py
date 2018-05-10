import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, target_variable=None, teacher_forcing_ratio=1):
        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_hidden=encoder_hidden,
                                           encoder_outputs=encoder_outputs,
                                           targets=target_variable,
                                           teacher_forcing_ratio=teacher_forcing_ratio)
        return seq_prob, seq_preds
