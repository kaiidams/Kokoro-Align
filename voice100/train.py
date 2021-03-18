# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence

BLANK_IDX = 0

class IndexArrayDataset:

    def __init__(self, file):
        with np.load(file) as f:
          self.indices = f['indices']
          self.data = f['data']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx - 1] if idx > 0 else 0
        end = self.indices[idx]
        return torch.from_numpy(self.data[start:end])

class TextAudioDataset(Dataset):
    def __init__(self, text_file, audio_file):
        self.text_dataset = IndexArrayDataset(text_file)
        self.audio_dataset = IndexArrayDataset(audio_file)
        assert len(self.text_dataset) == len(self.audio_dataset)

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, idx):
        text = self.text_dataset[idx]
        audio = self.audio_dataset[idx]
        return text, audio

class AudioToChar(nn.Module):

    def __init__(self, n_mfcc, hidden_dim, vocab_size):
        super(AudioToChar, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(n_mfcc, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, audio):
        lstm_out, _ = self.lstm(audio)
        lstm_out, lstm_out_len = pad_packed_sequence(lstm_out)
        return self.dense(lstm_out), lstm_out_len

def generate_batch(data_batch):
    text_batch, audio_batch = [], []
    for (text_item, audio_item) in data_batch:
        text_batch.append(text_item)
        audio_batch.append(audio_item)
    if False:
        text_batch = sorted(text_batch, key=lambda x: len(x), reverse=True)
        audio_batch = sorted(audio_batch, key=lambda x: len(x), reverse=True)
        text_batch = pack_sequence(text_batch)
        audio_batch = pack_sequence(audio_batch)
        return text_batch, audio_batch
    elif False:
        text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
        audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
        text_batch = pad_sequence(text_batch, BLANK_IDX)
        audio_batch = pad_sequence(audio_batch, BLANK_IDX)
        return text_batch, audio_batch, text_len, audio_len
    else:
        text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
        text_batch = pad_sequence(text_batch, BLANK_IDX)
        audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
        return text_batch, audio_batch, text_len

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (text, audio, text_len) in enumerate(dataloader):
        text = text.cuda()
        audio = audio.cuda()
        text_len = text_len.cuda()
        logits, probs_len = model(audio)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        text = text.transpose(0, 1)
        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = loss_fn(log_probs, text, probs_len, text_len)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(text)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #from .encoder import decode_text
        #print(decode_text(text))
        #print(audio.shape)

def test_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    test_loss = 0
    model.eval()
    for batch, (text, audio) in enumerate(dataloader):
        logits = model(audio)
        #print(logits.shape)
        text_lengths = torch.ones([text.shape[1]], dtype=torch.int32) * text.shape[0]
        audio_lengths = torch.ones([audio.shape[1]], dtype=torch.int32) * audio.shape[0]
        text = text.transpose(0, 1)
        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = loss_fn(logits, text, audio_lengths, text_lengths)

        test_loss += loss.item()

    test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")

def train(args):

    learning_rate = 0.001
    model = AudioToChar(n_mfcc=40, hidden_dim=128, vocab_size=29)
    loss_fn = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = loss_fn.cuda()
    model = model.cuda()

    ds = TextAudioDataset(
        text_file=f'data/{args.dataset}_text.npz',
        audio_file=f'data/{args.dataset}_audio.npz')
    train_ds, test_ds = torch.utils.data.random_split(ds, [len(ds) - len(ds) // 9, len(ds) // 9])

    train_dataloader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, collate_fn=generate_batch)
    test_dataloader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=generate_batch)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        #test_loop(test_dataloader, model, loss_fn, optimizer)
        PATH = './model/ctc.pth'
        torch.save(model.state_dict(), PATH)

def evaluate(args):
    from .encoder import decode_text, merge_repeated

    vocab_size = 29

    model = AudioToChar(n_mfcc=40, hidden_dim=128, vocab_size=vocab_size)
    model.eval()
    state = torch.load('./model/ctc.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state)

    ds = TextAudioDataset(
        text_file=f'data/{args.dataset}_text.npz',
        audio_file=f'data/{args.dataset}_audio.npz')
    train_ds, test_ds = torch.utils.data.random_split(ds, [len(ds) - len(ds) // 9, len(ds) // 9])
    test_dataloader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=generate_batch)
    for batch, (text, audio, text_len) in enumerate(test_dataloader):
        logits, logits_len = model(audio)
        # logits: [audio_len, batch_size, vocab_size]
        preds = torch.argmax(logits, axis=-1).T
        preds_len = logits_len
        for i in range(preds.shape[0]):
            pred_decoded = decode_text(preds[i, :preds_len[i]])
            pred_decoded = merge_repeated(pred_decoded)
            target_decoded = decode_text(text[:text_len[i], i])
            print('----')
            print(target_decoded)
            print(pred_decoded)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Split audio and encode with WORLD vocoder.')
    parser.add_argument('--eval', action='store_true', help='Split audio and encode with WORLD vocoder.')
    parser.add_argument('--dataset', default='css10ja', help='Analyze F0 of sampled data.')
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        raise ValueError('Unknown command')