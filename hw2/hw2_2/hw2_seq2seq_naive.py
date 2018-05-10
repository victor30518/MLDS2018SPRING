import time
start_a = time.time()
import random
import math
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *
import numpy as np
import os
import argparse
torch.backends.cudnn.enabled=True
USE_CUDA = True

parser= argparse.ArgumentParser(description='seq2seq')
parser.add_argument('--EVAL', action='store_true', help='Test the saved model')
parser.add_argument("--input_file", default=None)
parser.add_argument("--output_file", default=None)
parser.add_argument("--training_file", default=None)  #training data is not exist
parser.add_argument("--loadFilename", default='./50000_64.tar')
parser.add_argument("--n_iterations", default=120000, type=int)

args = parser.parse_args()

EVAL = False
if args.EVAL:
    EVAL = True
input_file = args.input_file
output_file = args.output_file
training_file = args.training_file
loadFilename = args.loadFilename
n_iterations = args.n_iterations

PAD_token = 0
BOS_token = 1
EOS_token = 2
UNK_token = 3

MIN_COUNT = 150
USE_CUDA = True
DICT_FILE = './word2index64.pkl'
DICT_FILE2 = './index2word64.pkl'
N_WORD = './n_word.txt'
save_dir = './model_naive'

MIN_LENGTH = 5
MAX_LENGTH = 26

attn_model = 'dot'
hidden_size = 512
n_layers = 2
dropout = 0.1
batch_size =64

clip = 50.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
iteration = 0
print_every = 500
evaluate_every = 2000
checkpoint_every = 2000

#build dictionary
class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4 # Count default tokens
        self.max_len = 0
        self.min_len = 1000

    def index_words(self, sentence):
        sen = sentence.split(' ')
        if len(sen) > self.max_len:   
            self.max_len = len(sen)
        if len(sen) < self.min_len:
            self.min_len = len(sen)
        for word in sen:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "BOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4 # Count default tokens

        for word in keep_words:
            self.index_word(word)


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return_indexs = []
    for i in sentence.split(' '):
        if i in lang.word2index.keys():
            return_indexs = return_indexs+[lang.word2index[i]]
        else:
            return_indexs = return_indexs+[UNK_token]

    return return_indexs + [EOS_token]

def remove_unk(lang, pairs):
    return_pairs=[]
    for pair in pairs:
        calls = np.array(indexes_from_sentence(lang, pair[0]))
        responses = np.array(indexes_from_sentence(lang, pair[1]))
        if len(calls[calls==3])<2 and len(responses[responses==3])<2: #check sentence has less than 3 unkown words 
            return_pairs+=[[pair[0], pair[1]]]
            
    return return_pairs   

def prepare_data(filename, reverse=False):
    print("Reading lines...")
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    share_lang = Lang('lang')
    c =[]
    for idx in range(len(lines)-1):
        if lines[idx]!="+++$+++" and lines[idx+1]!="+++$+++"         and len(lines[idx].split()) < MAX_LENGTH and len(lines[idx].split()) >= MIN_LENGTH         and len(lines[idx+1].split()) < MAX_LENGTH and len(lines[idx+1].split()) >= MIN_LENGTH:
            c+= [[lines[idx], lines[idx+1]]]
            share_lang.index_words(lines[idx])
            share_lang.index_words(lines[idx+1])
    # cut dict. for min count
    share_lang.trim(MIN_COUNT)
    
    # remove the sentences with more than 2 words
    c = remove_unk(share_lang, c)
    
    print('max sentence length: ', share_lang.max_len)
    print('Indexed %d words in corpus' % (share_lang.n_words))

    return share_lang, c

pairs = []
share_lang = Lang('lang')
if not EVAL:
    share_lang, pairs = prepare_data(training_file)

if DICT_FILE:
    share_lang.word2index = pickle.load(open(DICT_FILE, 'rb'))
    share_lang.index2word = pickle.load(open(DICT_FILE2, 'rb'))
    print('n_words:',share_lang.n_words)
    f = open(N_WORD,'r', encoding='utf-8')
    share_lang.n_words = int(f.read())
    f.close()
else:
    f = open("word2index64.pkl","wb")
    pickle.dump(share_lang.word2index,f)
    f.close()
    f = open("word2count64.pkl","wb")
    pickle.dump(share_lang.word2count,f)
    f.close()
    f = open("index2word64.pkl","wb")
    pickle.dump(share_lang.index2word,f)
    f.close()
    f = open('n_word.txt','w', encoding='utf-8')
    f.write(str(share_lang.n_words))
    f.close()

#Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


# batch data
def random_batch(batch_size):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(share_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(share_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        

    return input_var, input_lengths, target_var, target_lengths


# # Building the models

# ## The Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout)#, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)

        return outputs, hidden


# ## Attention Decoder
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        # self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input_seq, last_hidden):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)
        rnn_output = self.fc(rnn_output.squeeze(0)) # S=1 x B x N -> B x N

        output = self.out(rnn_output)
        return_output = self.softmax(output)

        return return_output, hidden


# # Training
# 
# ## Defining a training iteration
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([BOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()
    
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0]#, ec, dc


# Initialize models
encoder = EncoderRNN(share_lang.n_words, hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(hidden_size, share_lang.n_words, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

if loadFilename and EVAL:
    checkpoint = torch.load(loadFilename)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    encoder_optimizer.load_state_dict(checkpoint['en_opt'])
    decoder_optimizer.load_state_dict(checkpoint['de_opt'])
    iteration = checkpoint['iteration'] + 1
    loss_list = checkpoint['loss']
    print('training loss: ', loss_list[-1])
    
# Keep track of time elapsed and running averages
start = time.time()
print_loss_total = 0 # Reset every print_every

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# # Evaluating the network
# Evaluation is mostly the same as training, but there are no targets. Instead we always feed the decoder's predictions back to itself. Every time it predicts a word, we add it to the output string. If it predicts the EOS token we stop there. We also store the decoder's attention outputs for each step to display later.
def evaluate(input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq.split())]
    input_seqs = [indexes_from_sentence(share_lang, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
    
    if USE_CUDA:
        input_batches = input_batches.cuda()
        
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([BOS_token]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    # decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        # decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            #decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(share_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    
    return decoded_words#, decoder_attentions[:di+1, :len(encoder_outputs)]


# We can evaluate random sentences from the training set and print out the input, target, and output to make some subjective quality judgements:
def evaluate_randomly():
    [input_sentence, target_sentence] = random.choice(pairs)
    evaluate_and_show_attention(input_sentence, target_sentence)


# In[ ]:


def evaluate_and_show_attention(input_sentence, target_sentence=None):
    output_words = evaluate(input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)

#evaluation for testing data-----------------------------------------------    
# jieba.set_dictionary(jieba_dict)
def data_preprocessing(filename):
    return_inputs = []
    inputs = open(filename, 'r', encoding='utf-8').read().strip().split('\n')
    for i in inputs:
        #seg_list = jieba.cut(i.replace((' ','')), cut_all=False)
        #return_inputs.append(' '.join(seg_list))
        seg_list = i.split()
        return_inputs.append(' '.join(seg_list))
    return return_inputs



def evaluate_test(input_sentences, outputfilename):
    with open(outputfilename, 'w', encoding='utf-8') as f:
        for idx, sen in enumerate(input_sentences):
            output_words = evaluate(sen.strip())
            res = ''.join(output_words)
            f.write(res)
            f.write('\n')
    print('save the responses for input sentences !')


def evaluate_in_testing(filename, outputfilename):
    input_sentences = data_preprocessing(filename)
    evaluate_test(input_sentences, outputfilename)

# Begin!
loss_list =[]
if not EVAL:
    while iteration < n_iterations:
        iteration += 1
        
        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)

        # Run the train function
        loss= train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )

        # Keep track of loss
        print_loss_total += loss
        loss_list.append(loss)
        

        if iteration % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, iteration / n_iterations), iteration, iteration / n_iterations * 100, print_loss_avg)
            print(print_summary)
            


            
        #save checkpoints & plot loss
        if iteration % checkpoint_every == 0:
            directory = os.path.join(save_dir, 'model_new','{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss_list
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, batch_size)))

        if iteration % evaluate_every == 0:
            evaluate_randomly()          
else:
    print('start:',start_a)
    evaluate_in_testing(input_file, output_file)
    end_b = time.time()
    print('end:',end_b)
    print('time:' ,as_minutes(start_a-end_b))


