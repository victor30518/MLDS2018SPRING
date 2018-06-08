import argparse
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim=100

        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

generator = Generator()
if cuda:
    generator.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator.load_state_dict(torch.load('./model_500.pth'))
generator.eval()

latent_dim=100
np.random.seed(1)
z = Variable(Tensor(np.random.normal(0, 1, (25, latent_dim))))

gen_imgs = generator(z)
gen_imgs = (gen_imgs + 1) / 2 # rescale to (0,1)
#print(type(gen_imgs.data[0].cpu().numpy()))

import matplotlib.pyplot as plt
r, c = 5, 5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(gen_imgs.data[cnt].cpu().numpy().transpose(1,2,0))
        axs[i,j].axis('off')
        cnt += 1
fig.savefig("./samples/gan.png")
plt.close()
