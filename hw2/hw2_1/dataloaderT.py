import json

import random
import os
import h5py
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

        if self.mode == 'train':
            print('loaf label data')
            self.h5_label_file = h5py.File('info/data_label.h5', 'r', driver='core')

            #import visual feature
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

            self.seq_length = self.h5_label_file['labels'].shape[1]
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        elif self.mode == 'val':
            print('loaf label data')
            self.h5_label_file = h5py.File('info/data_test_label.h5', 'r', driver='core')

            #import visual feature
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

            self.seq_length = self.h5_label_file['labels'].shape[1]
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        elif self.mode == 'generate':
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

        else:
            fc_feat = self.video_feature[ix]

            label = np.zeros([self.seq_length], dtype='int')
            mask = np.zeros([self.seq_length], dtype='float32')
            # fetch the sequence labels
            ix1 = self.label_start_ix[ix]
            ix2 = self.label_end_ix[ix]
            # random select a caption for this video
            ixl = random.randint(ix1, ix2)
            label = self.h5_label_file['labels'][ixl]

            nonzero_ixs = np.nonzero(label)[0]
            mask[:nonzero_ixs.max() + 2] = 1

            data = {}
            data['fc_feats'] = torch.from_numpy(fc_feat)
            data['labels'] = torch.from_numpy(label)
            data['masks'] = torch.from_numpy(mask)
            data['ix'] = ix
            return data

    def __len__(self):
        if self.mode == 'val':
            return 100
        elif self.mode == 'train':
            return 1450
        elif self.mode == 'generate':
            return len(self.video_feature)
