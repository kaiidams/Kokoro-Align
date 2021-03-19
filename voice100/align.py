# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import numpy as np
from tqdm import tqdm
from voice100.encoder import encode_text2, decode_text2, merge_repeated2

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

def ctc_best_path(logits, labels, beam_size=8000, max_move=4):

    # Expand label with blanks
    tmp = labels
    labels = np.zeros(labels.shape[0] * 2 + 1, dtype=np.int32)
    labels[1::2] = tmp

    paths = [
        np.zeros([1], dtype=np.int32)
    ]
    alignments = [
        np.zeros([1], dtype=np.int32)
    ]
    score = np.zeros([1], dtype=np.float32)

    for i in tqdm(range(1, logits.shape[0])):

        alignment_min = (labels.shape[0] - beam_size) * i / logits.shape[0] - 10000
        alignment_max = (labels.shape[0] - beam_size) * i / logits.shape[0] + beam_size + 10

        next_path = np.zeros([max_move, labels.shape[0]], dtype=np.int32) - 1
        next_score = np.zeros([max_move, labels.shape[0]], dtype=np.float32) - 1e9

        for j in range(max_move):
            next_label_pos = alignments[-1] + j
            k, = np.nonzero((next_label_pos < labels.shape[0]) &
                (next_label_pos >= alignment_min)
                & (next_label_pos < alignment_max))            
            v = next_label_pos[k]
            next_path[j, v] = k
            next_score[j, v] = score[k] + logits[i, labels[v]]
            if j > 0 and j % 2 == 0:
                next_score[j, labels == 0] = -1e9

        k = np.argmax(next_score, axis=0)
        next_path = np.choose(k, next_path)
        next_score = np.choose(k, next_score)

        alignment, = np.nonzero(next_path >= 0)
        alignment = alignment.copy()
        paths.append(next_path[alignment].copy())
        score = next_score[alignment].copy()
        alignments.append(alignment)

    return get_path(paths, alignments, score)

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
    with open(f'data/{args.dataset}_transcript.txt') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            orig, text = parts
            origa.append((orig, text))
            text_end = text_start + (len(text.split(' ')) if text else 0)
            word_pos[text_start:text_end] = len(origa)
            text_start = text_end

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

            e = word_pos[text_start]
            m = word_pos[text_end]
            print(e, m)
            o = origa[e:m]
            o = ' '.join([x[0] for x in o])

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