# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import numpy as np
from tqdm import tqdm
from voice100.encoder import encode_text, decode_text, merge_repeated

class Voice100Dataset:
    def __init__(self, file):
        with np.load(file) as f:
            self.audio_indices = f['indices']
            self.audio_data = f['data']

    def __len__(self):
        return len(self.audio_indices)

    def __getitem__(self, index):
        audio_start = self.audio_indices[index - 1] if index else 0
        audio_end = self.audio_indices[index]
        audio = self.audio_data[audio_start:audio_end, :]
        return audio

def ctc_best_path2(logits, labels):
  # Expand label with blanks
  import numpy as np
  tmp = labels
  labels = np.zeros(labels.shape[0] * 2 + 1, dtype=np.int32)
  labels[1::2] = tmp

  cands = [
      (logits[0, labels[0]], [labels[0]])
  ]
  for i in tqdm(range(1, logits.shape[0])):
    next_cands = []
    for pos, (logit1, path1) in enumerate(cands):
      logit1 = logit1 + logits[i, labels[pos]]
      path1 = path1 + [labels[pos]]
      next_cands.append((logit1, path1))

    for pos, (logit2, path2) in enumerate(cands):
      if pos + 1 < len(labels):
        logit2 = logit2 + logits[i, labels[pos + 1]]
        path2 = path2 + [labels[pos + 1]]
        if pos + 1 == len(next_cands):
          next_cands.append((logit2, path2))
        else:
          logit, _ = next_cands[pos + 1]
          if logit2 > logit:
            next_cands[pos + 1] = (logit2, path2)
            
    for pos, (logit3, path3) in enumerate(cands):
      if pos + 2 < len(labels) and labels[pos + 1] == 0:
        logit3 = logit3 + logits[i, labels[pos + 2]]
        path3.append(labels[pos + 2])
        if pos + 2 == len(next_cands):
          next_cands.append((logit3, path3))
        else:
          logit, _ = next_cands[pos + 2]
          if logit3 > logit:
            next_cands[pos + 2] = (logit3, path3)
            
    cands = next_cands

  return cands[-1], 0
  
def ctc_best_path(logits, labels, beam_size=80000):

    def get_path(paths, alignments, score):
        s = []
        #k = np.arange(alignments[-1].shape[0])
        k = np.argmax(alignments[-1])
        k = np.array([k], dtype=np.int32)
        for path, alighment in zip(reversed(paths), reversed(alignments)):
            s.append(alighment[k])
            k = path[k]
        s = np.array(list(reversed(s))).T # (beam_size, audio_seq_len)
        si = np.argsort(score)[::-1]
        return s, score[k] #s[si], score[si]

    # Expand label with blanks
    import numpy as np
    tmp = labels
    labels = np.zeros(labels.shape[0] * 2 + 1, dtype=np.int32)
    labels[1::2] = tmp

    if False:
        print(decode_text(labels[:10]))

        print(logits.shape)
        for i in range(1000):
            x = logits[i]
            x = (np.exp(x) * 100)

            print(x.astype(int))
            #print(np.sum(x))
        hoge

    paths = [
        np.zeros([1], dtype=np.int32)
    ]
    alignments = [
        np.zeros([1], dtype=np.int32)
    ]
    score = np.zeros([1], dtype=np.float32)

    for i in tqdm(range(1, logits.shape[0])):

        alignment_min = (labels.shape[0] - beam_size) * i / logits.shape[0]
        alignment_max = (labels.shape[0] - beam_size) * i / logits.shape[0] + beam_size

        # Keep the current position
        path1 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score1 = np.zeros([labels.shape[0]], dtype=np.float32)-1e9
        k, = np.nonzero((alignments[-1] < labels.shape[0]) &
            (alignments[-1] >= alignment_min)
            & (alignments[-1] < alignment_max))            
        v = alignments[-1][k]
        path1[v] = k
        score1[v] = score[k] + logits[i, labels[v]]

        # Move forward
        path2 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score2 = np.zeros([labels.shape[0]], dtype=np.float32) - 1e9
        k, = np.nonzero((alignments[-1] + 1 < labels.shape[0]) &
            (alignments[-1] + 1 >= alignment_min)
            & (alignments[-1] + 1 < alignment_max))            
        v = (alignments[-1] + 1)[k]
        path2[v] = k
        score2[v] = score[k] + logits[i, labels[v]]

        # Skip blank and move forward
        path3 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score3 = np.zeros([labels.shape[0]], dtype=np.float32)-1e9
        k, = np.nonzero((alignments[-1] + 2 < labels.shape[0]) &
            (alignments[-1] + 2 >= alignment_min)
            & (alignments[-1] + 2 < alignment_max))            
        v = (alignments[-1] + 2)[k]
        path3[v] = k
        score3[v] = score[k] + logits[i, labels[v]]
        #print(path3[:10])
        #print(score3[:10])

        # Skip blank and move forward
        path4 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score4 = np.zeros([labels.shape[0]], dtype=np.float32)-1e9
        k, = np.nonzero((alignments[-1] + 3 < labels.shape[0]) &
            (alignments[-1] + 3 >= alignment_min)
            & (alignments[-1] + 3 < alignment_max))            
        v = (alignments[-1] + 3)[k]
        path4[v] = k
        score4[v] = score[k] + logits[i, labels[v]]

        k = (score1 > score2)
        score = np.where(k, score1, score2)
        path = np.where(k, path1, path2)
        #print('a')
        #print(path[:10])
        #print(score[:10])
        #print((score3 > score))
        #print(path.shape)
        #print(path3.shape)
        k = (score3 > score) & (labels != 0)
        score = np.where(k, score3, score)
        path = np.where(k, path3, path)

        k = (score4 > score) & (labels != 0)
        score = np.where(k, score4, score)
        path = np.where(k, path4, path)
        #print('b')
        #print(path[:10])
        #print(score[:10])
        #print(score)
        alignment, = np.nonzero(path >= 0)
        alignment = alignment.copy()
        paths.append(path[alignment].copy())
        score = score[alignment].copy()
        alignments.append(alignment)
        
        if True:
            if i % 1000 == 0:
                best_path, escore = get_path(paths, alignments, score)
                #k = np.argsort(score)[-1:]
                print('\n----')
                for p, s in zip(best_path, escore):
                    t = decode_text([labels[a] for a in p])
                    print(f'step: {i:4d} {s:.5f} {t}')
                    #print('alignment:', alignment.tolist())
                    #print('score:', score.tolist())

    return get_path(paths, alignments, score)

def phoneme(args):
    import re
    from glob import glob
    import torch
    sr = 22050
    from .train import AudioToChar

    model = AudioToChar(n_mfcc=40, hidden_dim=128, vocab_size=29)
    state = torch.load('./model/ctc.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()

    from .train import IndexArrayDataset
    ds = IndexArrayDataset('data/%s_audio.npz' % (args.dataset,))

    from .preprocess import open_index_data_for_write
    from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence

    with torch.no_grad():
        with open_index_data_for_write('data/%s_phoneme.npz' % (args.dataset,)) as file:
            with open('data/%s_phoneme.txt' % (args.dataset,), 'wt') as txtfile:
                for i, audio in enumerate(tqdm(ds)):
                    audio = pack_sequence([audio], enforce_sorted=False)
                    logits, logits_len = model(audio)
                    # logits: [audio_len, batch_size, vocab_size]
                    preds = torch.argmax(logits, axis=-1).T
                    preds_len = logits_len
                    pred_decoded = decode_text(preds[0, :preds_len[0]])
                    pred_decoded = merge_repeated(pred_decoded)
                    print(logits[:, 0, :].shape)
                    print(logits[:, 0, :].dtype)
                    file.write(logits[:, 0, :])
                    txtfile.write(f'{i+1}|{pred_decoded}\n')

def best_path(args):
    import torch
    from torch import nn
    with np.load('data/%s_phoneme.npz' % (args.dataset,)) as f:
        logits = f['data']
    
    print(np.min(logits))
    #with np.load('a.npz') as f:
    #    logits = f['logits']

    print(logits.shape)
    if False:
        k = logits[0].argmax(axis=-1)
        print(k.shape)
        k = decode_text(k)
        #print(k)
        #logits = np.concatenate([logits]*10, axis=1)
    logits = torch.from_numpy(logits)
    #print(np.min(logits.numpy()))
    print(logits.numpy())
    logits = logits * 10
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.numpy()

    s = ''
    with open('data/%s_transcript.txt' % (args.dataset)) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            s += parts[1]
    s = s.replace(' ', '')

    labels = encode_text(s)
    print(labels.shape)
    best_path, score = ctc_best_path(log_probs, labels)
    np.savez('data/%s_best_path.npz' % (args.dataset), best_path=best_path[0], score=score[0])
    l = decode_text([0 if x % 2 == 0 else labels[x // 2] for x in best_path[0]])
    print(l)

def align(args):

    s = ''
    with open('data/%s_transcript.txt' % (args.dataset)) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            s += parts[1]
    s = s.replace(' ', '')

    labels = encode_text(s)

    with np.load('data/%s_best_path.npz' % (args.dataset)) as f:
        best_path = f['best_path']

    file = 'data/%s_audio.npz' % (args.dataset,)
    with np.load(file) as f:
        audio_indices = f['indices']
        #audio_data = f['data']

    with open(f'data/{args.dataset}_alignment.txt', 'wt') as f:
        for i in range(len(audio_indices)):
            audio_start = audio_indices[i - 1] if i > 0 else 0 
            audio_end = audio_indices[i]
            text_start = best_path[audio_start]
            text_end = best_path[audio_end] if audio_end < len(best_path) else len(labels)
            
            text_start = (text_start) // 2
            text_end = (text_end) // 2

            s = labels[text_start:text_end]
            s = decode_text(s)
            f.write(f'{i+1}|{s}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phoneme', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--best_path', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--align', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', help='Dataset to process, css10ja, tsukuyomi_normal')
    parser.add_argument('--model_dir', default='model/ctc-20210313')

    args = parser.parse_args()

    if args.phoneme:
        phoneme(args)
    elif args.best_path:
        best_path(args)
    elif args.align:
        align(args)

if __name__ == '__main__':
    main()