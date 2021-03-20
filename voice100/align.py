# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import numpy as np
from tqdm import tqdm
from voice100.encoder import encode_text2, decode_text2, merge_repeated2

def get_path(beams, score):
    s = []
    cur_beam = np.argmax(beams[-1][0])
    cur_beam = np.array([cur_beam], dtype=np.int32)
    for label_pos, prev_beam in reversed(beams):
        s.append(label_pos[cur_beam])
        cur_beam = prev_beam[cur_beam]
    s = np.array(list(reversed(s))).T # (beam_size, audio_seq_len)
    si = np.argsort(score)[::-1]
    return s, score[cur_beam] #s[si], score[si]

def flush_determined_path(beams):
    cur_beam = np.arange(beams[-1][1].shape[0])
    i = len(beams) - 1
    while i >= 0:
        label_pos, prev_beam = beams[i]
        cur_beam = np.unique(prev_beam[cur_beam])
        i -= 1
        if (len(cur_beam) == 1):
            cur_beam = cur_beam[0]
            unique_end = i
            s = []
            while i >= 0:
                label_pos, prev_beam = beams[i]
                s.append(label_pos[cur_beam].item())
                cur_beam = prev_beam[cur_beam]
                i -= 1
            beams[:] = beams[unique_end + 1:]
            return list(reversed(s))

    return []

def ctc_best_path(log_probs, labels, beam_size=1000, max_move=4):

    # Expand label with blanks
    tmp = labels
    labels = np.zeros(labels.shape[0] * 2 + 1, dtype=np.int32)
    labels[1::2] = tmp

    labels_len = labels.shape[0]
    log_probs_len = log_probs.shape[0]

    print(f"Label length: {labels_len}")
    print(f"Time length: {log_probs_len}")

    beams = []
    label_pos = np.zeros([1], dtype=np.int32)
    score = np.zeros([1], dtype=np.float32)

    determined_path = []

    for i in tqdm(range(0, log_probs_len)):

        next_beam = np.zeros([max_move, labels_len], dtype=np.int32) - 1
        next_score = np.zeros([max_move, labels_len], dtype=np.float32) - np.inf

        next_label_pos_min = max(0, labels_len * i / log_probs_len - beam_size // 2)
        next_label_pos_max = min(next_label_pos_min + beam_size, labels_len)

        for j in range(max_move):
            next_label_pos = label_pos + j
            k, = np.nonzero(
                (next_label_pos >= next_label_pos_min)
                & (next_label_pos < next_label_pos_max))            
            v = next_label_pos[k]
            next_beam[j, v] = k
            next_score[j, v] = score[k] + log_probs[i, labels[v]]

            # Don't move from one blank to another blank.
            if j > 0 and j % 2 == 0:
                next_score[j, labels == 0] = -np.inf

        k = np.argmax(next_score, axis=0)
        next_beam = np.choose(k, next_beam)
        next_score = np.choose(k, next_score)

        label_pos, = np.nonzero(next_beam >= 0)
        label_pos = label_pos.copy()
        score = next_score[label_pos].copy()

        beams.append((
            label_pos,
            next_beam[label_pos].copy()
            ))

        #if i % 10 == 0:
        #    hist[i // 10, :] = next_score
        if i % 1000 == 0:
            determined_path.extend(flush_determined_path(beams))
            print(len(determined_path))

    beams.append((
        np.array([labels_len], dtype=np.int32),
        np.expand_dims(np.argmax(beams[-1][0]), axis=0)))
    determined_path.extend(flush_determined_path(beams))
    print(len(determined_path))

    return np.array(determined_path, dtype=np.int32)

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

    best_path = ctc_best_path(log_probs, labels)
    np.savez('data/%s_best_path.npz' % (args.dataset), best_path=best_path)
    l = decode_text2([0 if x % 2 == 0 else labels[x // 2] for x in best_path])
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

            if text_start >= len(orig_pos):
                o = ''
            else:
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