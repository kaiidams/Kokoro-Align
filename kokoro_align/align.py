# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import os
from tqdm import tqdm
from kokoro_align.encoder import decode_text, merge_repeated


def get_path(beams, score):
    s = []
    cur_beam = np.argmax(beams[-1][0])
    cur_beam = np.array([cur_beam], dtype=np.int32)
    for label_pos, prev_beam in reversed(beams):
        s.append(label_pos[cur_beam])
        cur_beam = prev_beam[cur_beam]
    s = np.array(list(reversed(s))).T  # (beam_size, audio_seq_len)
    # si = np.argsort(score)[::-1]
    return s, score[cur_beam]  # s[si], score[si]


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

        next_label_pos_min = max(0, labels_len * i // log_probs_len - beam_size // 2)
        next_label_pos_max = min(next_label_pos_min + beam_size, labels_len)

        next_beam = np.zeros([max_move, next_label_pos_max - next_label_pos_min], dtype=np.int32) - 1
        next_score = np.zeros([max_move, next_label_pos_max - next_label_pos_min], dtype=np.float32) - np.inf

        for j in range(max_move):
            next_label_pos = label_pos + j
            k, = np.nonzero(
                (next_label_pos >= next_label_pos_min)
                & (next_label_pos < next_label_pos_max))
            v = next_label_pos[k]
            next_beam[j, v - next_label_pos_min] = k
            next_score[j, v - next_label_pos_min] = score[k] + log_probs[i, labels[v]]

            # Don't move from one blank to another blank.
            if j > 0 and j % 2 == 0:
                next_score[j, labels[next_label_pos_min:next_label_pos_max] == 0] = -np.inf

        k = np.argmax(next_score, axis=0)
        next_beam = np.choose(k, next_beam)
        next_score = np.choose(k, next_score)

        label_pos, = np.nonzero(next_beam >= 0)
        label_pos = label_pos.copy()
        score = next_score[label_pos].copy()
        beam = next_beam[label_pos].copy()
        label_pos += next_label_pos_min

        beams.append((label_pos, beam))

        if i % 10000 == 0:
            determined_path.extend(flush_determined_path(beams))
            # print(len(determined_path))

    beams.append((
        np.array([labels_len], dtype=np.int32),
        np.expand_dims(np.argmax(beams[-1][0]), axis=0)))
    determined_path.extend(flush_determined_path(beams))
    # print(len(determined_path))

    best_path = np.array(determined_path, dtype=np.int32)
    best_labels = labels[best_path]
    best_scores = log_probs[np.arange(best_labels.shape[0]), best_labels]

    return best_path, best_labels, best_scores


def best_path(input_file, voca_file, output_file):
    with np.load(input_file) as f:
        logits = f['data']

    logits = logits - np.mean(logits, axis=-1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    from .transcript import read_transcript
    labels = read_transcript(voca_file)

    best_path, best_labels, best_scores = ctc_best_path(log_probs, labels)
    np.savez(
        output_file, best_path=best_path,
        best_labels=best_labels, best_scores=best_scores)


def align(best_path_file, mfcc_file, voca_file, align_file, remove_wordsep):

    from .transcript import VocaAligner

    with np.load(best_path_file) as f:
        best_path = f['best_path']
        best_labels = f['best_labels']
        best_scores = f['best_scores']
    best_path = best_path // 2

    with np.load(mfcc_file) as f:
        audio_indices = f['indices']

    aligner = VocaAligner(voca_file)

    text_start = 0
    # origa = []
    # word_pos = np.zeros([best_path.shape[0]], dtype=np.int32)

    try:
        with open(align_file, 'wt') as f:
            for i in range(len(audio_indices)):
                audio_start = audio_indices[i - 1] if i > 0 else 0
                audio_end = audio_indices[i]
                text_start = best_path[audio_start]
                text_end = best_path[audio_end] if audio_end < len(best_path) else len(aligner)
                text_start = min(text_start, len(aligner))
                text_end = min(text_end, len(aligner))

                labels = best_labels[audio_start:audio_end]
                scores = best_scores[audio_start:audio_end]

                decoded = merge_repeated(decode_text(labels))
                non_blanks = np.sum(labels != 0).item()
                non_blanks_score = np.sum(scores[labels != 0]).item()
                all_score = np.sum(scores).item()

                text, voca = aligner.get_token(text_start, text_end, remove_wordsep=remove_wordsep)

                f.write(f'{audio_end}|{text}|{voca}|{decoded}|{non_blanks}|{non_blanks_score}|{all_score}\n')
    except:
        os.unlink(align_file)
        raise


def pandas_read_align(files):
    import pandas as pd
    d = []
    for file in files:
        with open(file) as f:
            audio_end = '0'
            for line in f:
                parts = line.rstrip().split('|')
                parts.insert(0, audio_end)
                d.append(parts)
                audio_end = parts[1]
    df = pd.DataFrame(d, columns='audio_start audio_end text voca decoded non_blanks non_blanks_score all_score'.split(),)
    df['audio_start'] = df['audio_start'].astype(int)
    df['audio_end'] = df['audio_end'].astype(int)
    df['non_blanks'] = df['non_blanks'].astype(int)
    df['non_blanks_score'] = df['non_blanks_score'].astype(float)
    df['all_score'] = df['all_score'].astype(float)
    df['audio_len'] = df['audio_end'] - df['audio_start']
    return df
