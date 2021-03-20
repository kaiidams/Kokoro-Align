# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import numpy as np
from tqdm import tqdm
from voice100.encoder import encode_text2, decode_text2, merge_repeated2

def get_path(prev_beam, label_pos, score):
    s = []
    #k = np.arange(label_pos[-1].shape[0])
    k = np.argmax(label_pos[-1])
    k = np.array([k], dtype=np.int32)
    for path, alighment in zip(reversed(prev_beam), reversed(label_pos)):
        s.append(alighment[k])
        k = path[k]
    s = np.array(list(reversed(s))).T # (beam_size, audio_seq_len)
    si = np.argsort(score)[::-1]
    return s, score[k] #s[si], score[si]

def ctc_best_path(log_probs, labels, beam_size=2000, max_move=4):

    num_labels = labels.shape[0]
    num_log_probs = log_probs.shape[0]

    # Expand label with blanks
    tmp = labels
    labels = np.zeros(num_labels * 2 + 1, dtype=np.int32)
    labels[1::2] = tmp

    prev_beam = [
        np.zeros([1], dtype=np.int32)
    ]
    label_pos = [
        np.zeros([1], dtype=np.int32)
    ]
    score = np.zeros([1], dtype=np.float32)

    for i in tqdm(range(1, num_log_probs)):

        label_pos_min = (num_labels - beam_size) * i / num_log_probs - 10
        label_pos_max = (num_labels - beam_size) * i / num_log_probs + beam_size + 10

        next_path = np.zeros([max_move, num_labels], dtype=np.int32) - 1
        next_score = np.zeros([max_move, num_labels], dtype=np.float32) - 1e9

        for j in range(max_move):
            next_label_pos = label_pos[-1] + j
            k, = np.nonzero((next_label_pos < num_labels) &
                (next_label_pos >= label_pos_min)
                & (next_label_pos < label_pos_max))            
            v = next_label_pos[k]
            next_path[j, v] = k
            next_score[j, v] = score[k] + log_probs[i, labels[v]]
            if j > 0 and j % 2 == 0:
                next_score[j, labels == 0] = -1e9

        k = np.argmax(next_score, axis=0)
        next_path = np.choose(k, next_path)
        next_score = np.choose(k, next_score)

        alignment, = np.nonzero(next_path >= 0)
        alignment = alignment.copy()
        prev_beam.append(next_path[alignment].copy())
        score = next_score[alignment].copy()
        label_pos.append(alignment)

    return get_path(prev_beam, label_pos, score)

def best_path(args):
    import torch
    from torch import nn
    with np.load(f'data/{args.dataset}_logits.npz') as f:
        logits = f['data']

    logits = torch.from_numpy(logits)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.numpy()

    from .transcript import read_transcript
    from voice100.encoder import encode_text2, decode_text2, merge_repeated2
    s = read_transcript(args.dataset)
    labels = encode_text2(s)
    print(labels.shape)

    best_path, score = ctc_best_path(log_probs, labels)
    np.savez('data/%s_best_path.npz' % (args.dataset), best_path=best_path[0], score=score[0])
    l = decode_text2([0 if x % 2 == 0 else labels[x // 2] for x in best_path[0]])
    print(l)

def align(args):

    from .transcript import read_transcript
    from voice100.encoder import encode_text2, decode_text2, merge_repeated2
    s = read_transcript(args.dataset)
    labels = encode_text2(s)

    with np.load('data/%s_best_path.npz' % (args.dataset)) as f:
        best_path = f['best_path']

    file = 'data/%s_audio.npz' % (args.dataset,)
    with np.load(file) as f:
        audio_indices = f['indices']
        #audio_data = f['data']

    text_start = 0
    origa = []
    word_pos = np.zeros([best_path.shape[0]], dtype=np.int32)

    orig_tokens = []
    text_tokens = []
    orig_pos = []
    pos = 0
    with open(f'data/{args.dataset}_transcript.txt') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            orig, text = parts
            orig_tokens.append(orig)
            text_tokens.append(text)

            text_len = len(encode_text2(text))
            orig_pos.extend([pos] * text_len)
            pos += 1

    with open(f'data/{args.dataset}_alignment.txt', 'wt') as f:
        for i in range(len(audio_indices)):
            audio_start = audio_indices[i - 1] if i > 0 else 0 
            audio_end = audio_indices[i]
            text_start = best_path[audio_start]
            text_end = best_path[audio_end] if audio_end < len(best_path) else len(labels) * 2 + 1
            
            text_start = (text_start) // 2
            text_end = (text_end) // 2

            s = labels[text_start:text_end]
            s = decode_text2(s)

            orig_start = orig_pos[text_start]
            orig_end = orig_pos[text_end] if text_end < len(labels) else len(orig_tokens)
            o = orig_tokens[orig_start:orig_end]
            o = ' '.join(o)

            f.write(f'{i+1}|{o}|{s}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_path', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--align', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', help='Dataset to process, css10ja, tsukuyomi_normal')

    args = parser.parse_args()

    if args.best_path:
        best_path(args)
    elif args.align:
        align(args)
    else:
        raise ValueError("Unknown command")

if __name__ == '__main__':
    main()