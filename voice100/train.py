import pickle
import torch
import random
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import shutil
import math

from .model import Voice100Model
from .dataset import Voice100Dataset

train_data = Voice100Dataset('data/css10ja_train.npz')
val_data = Voice100Dataset('data/css10ja_val.npz')

bptt = None

def make_padding_mask(max_len, lens):
    padding_mask = torch.arange(max_len).expand(len(lens), max_len) >= torch.tensor(lens).unsqueeze(1)
    return padding_mask

def get_batch(data, start=-1):
    inputs = []
    targets = []
    for i in range(bptt):
        if start >= 0:
            idx = start + i
            if idx >= len(data):
                break
        else:
            idx = random.randrange(0, len(data))
        _, text, audio = data[idx]
        inputs.append(text)
        targets.append(audio)

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

ntokens = 28 # the size of vocabulary
emsize = 200 # embedding dimension
vosize = 27 # Vocoder dimension
nhid = 400 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = Voice100Model(ntokens, emsize, vosize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.MSELoss(reduction='none')

epochs = 20 # The number of epochs

def get_previous_output(output):
    return torch.cat([
        torch.zeros_like(output[:1, :, :]),
        output[:-1, :, :]], axis=0)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    total_steps = 1000
    loss_weights = torch.ones([1, 1, vosize], dtype=torch.float32).to(device)
    loss_weights[0, 0, 26] = 5.0
    for batch, i in enumerate(range(0, total_steps)):
        inputs, targets, inputs_padding_mask, targets_padding_mask = get_batch(train_data, i)
        inputs = torch.transpose(inputs, 0, 1)
        targets = torch.transpose(targets, 0, 1)
        targets_mask = model.generate_square_subsequent_mask(targets.shape[0]).to(device)
        previous_output = get_previous_output(targets)
        optimizer.zero_grad()
        output = model(inputs, previous_output, inputs_padding_mask, targets_mask)

        loss = criterion(output, targets) * loss_weights
        #loss = torch.max(loss, axis=2).values
        loss = torch.mean(loss, axis=2)
        loss = torch.transpose(loss, 0, 1)
        loss = loss[torch.logical_not(targets_padding_mask)]
        loss = torch.sum(loss) / torch.sum(torch.logical_not(targets_padding_mask).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 100
        if batch % log_interval == log_interval - 1:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:3.5f}'.format(
                    epoch, batch + 1, total_steps, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            if math.isnan(total_loss):
                raise ValueError() 
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    n = 0
    loss_weights = torch.ones([1, 1, vosize], dtype=torch.float32).to(device)
    with torch.no_grad():
        for i in range(0, len(data_source['id']), bptt):
            inputs, targets, inputs_padding_mask, targets_padding_mask = get_batch(data_source, i)
            inputs = torch.transpose(inputs, 0, 1)
            targets = torch.transpose(targets, 0, 1)
            previous_output = get_previous_output(targets)
            targets_mask = model.generate_square_subsequent_mask(targets.shape[0]).to(device)
            output = eval_model(inputs, previous_output, inputs_padding_mask, targets_mask)

            loss = criterion(output, targets) * loss_weights
            #loss = torch.max(loss, axis=2).values
            loss = torch.mean(loss, axis=2)
            loss = torch.transpose(loss, 0, 1)
            loss = loss[torch.logical_not(targets_padding_mask)]

            total_loss += torch.sum(loss)
            n += loss.shape[0]
    return total_loss / n


def main_worker(args):
    global epoch
    global optimizer
    global scheduler

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    #epochs = 20
    #model.load_state_dict(torch.load(f'model/ckpt_{epochs}.pt', map_location=device))

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid loss {:3.5f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss))
        print('-' * 89)

        checkpoint_path = os.path.join(args.save_dir, f'ckpt_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict()
        }, checkpoint_path)

        if val_loss < best_val_loss:
            best_checkpoint_path = os.path.join(args.save_dir, 'ckpt_best.pt')
            shutil.copyfile(checkpoint_path, best_checkpoint_path)
            best_val_loss = val_loss

        scheduler.step()

def test(args):
    from .preprocess import writewav, text2feature, feature2wav
    from tqdm import tqdm
    checkpoint_path = os.path.join(args.save_dir, 'ckpt_best.pt')
    checkpoint = torch.load(f'model/simple/ckpt_best.pt', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        monophone = 'koNnichiwa,ohayo:gozaimasu.'
        monophone = 'iqtaiimanowakaimonowa,karadamedane.kudaraNbyo:kibakarishite.'
        inputs = torch.tensor(text2feature(monophone), dtype=torch.long)
        inputs = inputs.unsqueeze(1)
        #inputs = torch.cat([inputs, torch.zeros([60, 1], dtype=torch.long)], axis=0)
        inputs = inputs.to(device)
        print(inputs)
        targets = torch.zeros([1, 1, vosize], dtype=torch.float32).to(device)
        inputs_padding_mask = torch.zeros([1, inputs.shape[0]], dtype=torch.bool).to(device)
        for i in tqdm(range(300)):
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

def main():
    global bptt
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--save_dir', help='path to save checkpoints', required=True)
    args = parser.parse_args()
    bptt = args.batch_size
    if args.train:
        main_worker(args)
    elif args.test:
        test(args)
    else:
        raise ValueError()

if __name__ == '__main__':
    main()