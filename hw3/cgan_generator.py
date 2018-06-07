import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import skimage.io
import scipy.misc 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import sys
import pickle
from collections import defaultdict

test_tag_path = sys.argv[1]

# 12個
color_hair = ['aqua hair', 'black hair', 'blonde hair', 'blue hair',
              'brown hair', 'gray hair', 'green hair', 'orange hair',
              'pink hair', 'purple hair', 'red hair', 'white hair']
# 11個
color_eyes = ['aqua eyes', 'black eyes', 'blue eyes',
              'brown eyes', 'gray eyes', 'green eyes', 'orange eyes',
              'pink eyes', 'purple eyes', 'red eyes', 'yellow eyes']

def pair2id(pair_list):
    id_list = []
    for one_pair in pair_list:
        label_id = color_hair.index(one_pair[0]) * len(color_eyes) + color_eyes.index(one_pair[1])
        id_list.append(label_id)
    return id_list

def load_test_tag(path):
    y_tag_pair = []
    cnt = 0
    with open(path,"r") as file:
        tags_list = file.readlines()
        for one_tags in tags_list:
            img_id = one_tags.split(",")[0]
            tag_pairs = one_tags.strip().split(",")[1].split(" ")
            hair_tag = tag_pairs[0] + " " + tag_pairs[1]
            eyes_tag = tag_pairs[2] + " " + tag_pairs[3]
            y_tag_pair.append([color_hair.index(hair_tag),color_eyes.index(eyes_tag)])
            cnt += 1
        y_train = y_tag_pair
    return np.array(y_train)

if not os.path.exists('./samples'):
	os.mkdir('./samples')

class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(123, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))/2.0+0.5
        return x

def save_imgs(generator,labels):
    r, c = 5, 5
    torch.manual_seed(7)
    noise = torch.randn((r*c, 100)).view(-1, 100, 1, 1)
    noise = Variable(noise.cuda())
    
    labels = torch.from_numpy(labels)
    torch_labels = torch.cat([hair_onehot[labels[:,0]],eyes_onehot[labels[:,1]]],1)
    torch_labels = Variable(torch_labels.cuda())

    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator(noise,torch_labels)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:].cpu().data.numpy().transpose(1, 2, 0))
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./samples/cgan.png")
    plt.close()

# load model
G = generator()
G.cuda()
G.load_state_dict(torch.load("model/cgan_generator_model.pkl"))
print("model loaded")

# load testing data
test_labels = load_test_tag(test_tag_path)

# label preprocess
hair_onehot = torch.zeros(12, 12)
hair_onehot = hair_onehot.scatter_(1, torch.LongTensor(list(range(12))).view(12,1), 1).view(12, 12, 1, 1)

eyes_onehot = torch.zeros(11, 11)
eyes_onehot = eyes_onehot.scatter_(1, torch.LongTensor(list(range(11))).view(11,1), 1).view(11, 11, 1, 1)

G.eval()

# generator image
save_imgs(G,test_labels)
print("Generate cgan_original.png")