import json

import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, mode, folder_path=None):
        self.mode = mode  # train/val/generate

        # load the json file of dictionary(word_to_id and inverse)
        info = json.load(open('./info/info.json'))
        self.ix_to_word = info['ix_to_word'] #len(self.ix_to_word) is vocab size(1961)

        print('Data loading...')

        if self.mode == 'generate':
            print('load video features')
            file_path = os.path.join(folder_path, 'id.txt')
            video_id=[]
            fp = open(file_path, "r")
            for line in fp:
                video_id.append(line.strip())
            fp.close()

            dataPath = os.path.join(folder_path, 'feat/')
            self.video_feature = []
            for name in video_id:
                self.video_feature.append(np.load(dataPath+name+".npy").astype("float32")) 

    def __getitem__(self, ix):
        if self.mode == 'generate':
            fc_feat = self.video_feature[ix]
            data = {}
            data['fc_feats'] = torch.from_numpy(fc_feat)
            data['ix'] = ix
            return data


    def __len__(self):
        if self.mode == 'generate':
            return len(self.video_feature)
