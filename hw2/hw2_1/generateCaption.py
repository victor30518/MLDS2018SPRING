import json
import os
import sys
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import Encoder, Decoder, Model
from dataloader import VideoDataset

# folder_path = '/media/huangyunghan/7d1a973c-6589-45e4-821d-a0f50d965fd7/MLDS_hw2_1_data/testing_data/'
# output_path = 'result_test.txt'
folder_path = sys.argv[1]
output_path = sys.argv[2]

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    decoded_seq = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].data.cpu().numpy()[0]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        decoded_seq.append(txt)
    return decoded_seq

def test(model, dataset, vocab):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    results = []
    for data in loader:
        fc_feats = Variable(data['fc_feats']).cuda()

        seq_probs, seq_preds = model(fc_feats, teacher_forcing_ratio=0)
        sents = decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            results.append(sent)

    file_path = os.path.join(folder_path, 'id.txt')
    video_id=[]
    fp = open(file_path, "r")
    for line in fp:
        video_id.append(line.strip())
    fp.close()

    f = open(output_path,'w')
    for i in range(len(results)):
        f.write(video_id[i] + ',' + results[i]+'\n')
    f.close()

dim_vid = 4096
dim_hidden = 512
dim_word = 512

dataset = VideoDataset('generate', folder_path)
vocab_size = dataset.get_vocab_size()
seq_length = 25


encoder = Encoder(dim_vid, dim_hidden)
decoder = Decoder(vocab_size, seq_length, dim_hidden, dim_word, rnn_dropout_p=0.2)
model = Model(encoder, decoder).cuda()

model = nn.DataParallel(model)
model.load_state_dict(torch.load('./model_68.pth'))
model.eval()
test(model, dataset, dataset.get_vocab())



