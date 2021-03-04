import pickle
import torch
import random
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import math

from model import TransformerModel

with open('data/train_tsukuyomi_normal.pkl', 'rb') as f:
    train_data = pickle.load(f)

bptt = 5

def make_padding_mask(max_len, lens):
    padding_mask = torch.arange(max_len).expand(len(lens), max_len) >= torch.tensor(lens).unsqueeze(1)
    return padding_mask

def get_batch(train_data, i):
    inputs = []
    targets = []
    for _ in range(bptt):
        idx = random.randrange(0, len(train_data['id']))
        inputs.append(train_data['text'][idx])
        targets.append(train_data['audio'][idx])

    inputs_len = np.array([len(x) for x in inputs], dtype=np.int)
    targets_len = np.array([x.shape[0] for x in targets], dtype=np.int)

    max_inputs_len = np.max(inputs_len)
    max_targets_len = np.max(targets_len)

    inputs = np.vstack([
        np.concatenate([x, np.zeros(max_inputs_len - len(x), dtype=np.int)])
        for x in inputs
    ])
    targets = np.array([
        np.concatenate([x, np.zeros([max_targets_len - len(x), vosize], dtype=np.float)], axis=0)
        for x in targets
    ])

    inputs = torch.tensor(inputs, dtype=torch.long).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    inputs_padding_mask = make_padding_mask(max_inputs_len, inputs_len).to(device)
    targets_padding_mask = make_padding_mask(max_targets_len, targets_len).to(device)

    return inputs, targets, inputs_padding_mask, targets_padding_mask

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ntokens = 25 # the size of vocabulary
emsize = 200 # embedding dimension
vosize = 27 # Vocoder dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, vosize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.MSELoss(reduction='none')
lr = 1.0 # 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs = 20 # The number of epochs

import time
def train_step():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    total_steps = 100
    for batch, i in enumerate(range(0, total_steps)):
        inputs, targets, inputs_padding_mask, targets_padding_mask = get_batch(train_data, i)
        inputs = torch.transpose(inputs, 0, 1)
        targets = torch.transpose(targets, 0, 1)
        e0 = torch.cat([torch.zeros_like(targets[:1, :, :]), targets[:-1, :, :]], axis=0)
        targets_mask = model.generate_square_subsequent_mask(targets.shape[0]).to(device)
        optimizer.zero_grad()
        output = model(inputs, e0, inputs_padding_mask, targets_mask)
        if False:
            #print(targets.shape)
            #targets_mask = torch.zeros([targets.shape[0], targets.shape[0]], dtype=torch.float32)
            #print(inputs)
            #print(targets)
            #print(max_inputs_len)
            #print(inputs.shape)
            #print(e0)
            k = targets * torch.transpose(targets_mask.unsqueeze(2), 0, 1)
            k = torch.sum(k, axis=1)
            k = torch.sum(k, axis=0)
            k = k / torch.sum(targets_mask)
            print(torch.min(targets))
            print(output[50,:, :])

        loss = torch.max(criterion(output, targets), axis=2).values
        loss = loss * torch.transpose(1.0 - targets_padding_mask.float(), 0, 1)
        loss = torch.sum(loss) / torch.sum(1.0 - targets_padding_mask.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 10 #200
        if batch % log_interval == log_interval - 1:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                    epoch, batch + 1, total_steps, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    global epoch
    best_val_loss = float("inf")
    best_model = None
    os.makedirs('model', exist_ok=True)
    #torch.load(model.state_dict(), f'model/ckpt_4.pt')

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_step()
        val_loss = 0#evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        torch.save(model.state_dict(), f'model/ckpt_{epoch}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

def test():
    from preprocess import writewav, text2feature, feature2wav
    from tqdm import tqdm

    model.load_state_dict(torch.load(f'model/ckpt_{epochs}.pt', map_location=device))
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(text2feature('koNnichiwaohayo:gozaimasu'), dtype=torch.long)
        inputs = inputs.unsqueeze(1)
        #inputs = torch.cat([inputs, torch.zeros([60, 1], dtype=torch.long)], axis=0)
        inputs = inputs.to(device)
        print(inputs)
        targets = torch.zeros([1, 1, vosize], dtype=torch.float32).to(device)
        inputs_padding_mask = torch.zeros([1, inputs.shape[0]], dtype=torch.bool).to(device)
        for i in tqdm(range(100)):
            #targets_mask = torch.zeros([targets.shape[0], targets.shape[0]], dtype=torch.float32)
            targets_mask = model.generate_square_subsequent_mask(targets.shape[0]).to(device)
            output = model(inputs, targets, inputs_padding_mask, targets_mask)
            targets = torch.cat([
                targets,
                output[-1:]
            ], axis=0)

    targets = targets[1:, 0, :].cpu().numpy()
    y = feature2wav(targets)
    y = y * (0.8 / y.max())
    writewav('data/test.wav', y, 16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.train:
        train()
    elif args.test:
        test()
    else:
        raise ValueError()