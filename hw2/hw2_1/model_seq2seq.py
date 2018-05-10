import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
import numpy as np
import os
import json

from models import Encoder, Decoder, Model
from dataloaderT import VideoDataset

folder_path = '/media/huangyunghan/7d1a973c-6589-45e4-821d-a0f50d965fd7/MLDS_hw2_1_data/training_data/'
folder_path_val = '/media/huangyunghan/7d1a973c-6589-45e4-821d-a0f50d965fd7/MLDS_hw2_1_data/testing_data/'

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        loss_fn = nn.NLLLoss(reduce=False)
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = to_contiguous(logits).view(-1, logits.shape[2])
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)
        loss = loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def val(dataloader, model, crit):
    model.eval()

    losses = []
    for data in dataloader:
        torch.cuda.synchronize()
        fc_feats = Variable(data['fc_feats']).cuda()
        labels = Variable(data['labels']).long().cuda()
        masks = Variable(data['masks']).cuda()
        seq_probs, predicts = model(fc_feats, labels)
        loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
        val_loss = loss.data[0]
        losses.append(val_loss)
    val_loss = sum(losses) / len(losses)
    return val_loss


def train(train_loader, val_loader, model, crit, optimizer, lr_scheduler, epochs):
    model.train()
    model = nn.DataParallel(model)
    # lowest val loss
    best_loss = None
    for epoch in range(epochs):
        lr_scheduler.step()

        iteration = 0

        for data in train_loader:
            torch.cuda.synchronize()
            fc_feats = Variable(data['fc_feats']).cuda()
            labels = Variable(data['labels']).long().cuda()
            masks = Variable(data['masks']).cuda()
            
            seq_probs, predicts = model(fc_feats, labels, teacher_forcing_ratio=0.9)
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, grad_clip=0.1)
            optimizer.step()
            train_loss = loss.data[0]
            torch.cuda.synchronize()
            iteration += 1

            if iteration==1:
                print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, train_loss))

        if epoch % 50 == 0:
            checkpoint_path = './ckpt/model_%d.pth' % (epoch)
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to %s" % (checkpoint_path))
            val_loss = val(val_loader, model, crit)
            print("Val loss is: %.6f" % (val_loss))
            model.train()
            if best_loss is None or val_loss < best_loss:
                print("(epoch %d), now lowest val loss is %.6f" % (epoch, val_loss))
                checkpoint_path = './ckpt/model_best.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print("best model saved to %s" % (checkpoint_path))
                best_loss = val_loss


dim_vid = 4096
dim_hidden = 512
dim_word = 512
epochs = 151

train_dataset = VideoDataset('train', folder_path)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

vocab_size = train_dataset.get_vocab_size()
seq_length = train_dataset.seq_length

val_dataset = VideoDataset('val', folder_path_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)


encoder = Encoder(dim_vid, dim_hidden)
decoder = Decoder(vocab_size, seq_length, dim_hidden, dim_word, rnn_dropout_p= 0.2)
model = Model(encoder, decoder).cuda()

crit = LanguageModelCriterion()

optimizer = optim.Adam(model.parameters(), lr= 4e-4, weight_decay= 0)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

train(train_dataloader, val_dataloader, model, crit, optimizer, exp_lr_scheduler, epochs)


