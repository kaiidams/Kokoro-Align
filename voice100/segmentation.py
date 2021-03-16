# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import numpy as np
from tqdm import tqdm
from voice100.encoder import encode_text, decode_text

def test1():
    from voice100._text2voca import text2voca
    from bs4 import BeautifulSoup

    s = ''

    def _process_soup(node):
        global s

        def g(text):
            text = text.strip()
            if text:
                try:
                    yomi = text2voca(text, ignore_error=True)
                    f.write('%s|%s\n' % (text, yomi))
                except:
                    print(text)

        if node.name:
            if node.name == 'ruby':
                rb = ''.join(child.string for child in node.find_all('rb') if child.string)
                rt = ''.join(child.string for child in node.find_all('rt') if child.string)
                s += rb
            elif node.name == 'br':
                g(s)
                s = ''    
            else:
                if node.name in ['div', 'h1', 'h2', 'h3']:
                    g(s)
                    s = ''    
                for child in node.children:
                    _process_soup(child)
        else:
            s += node

def test1():
    global f

    with open('data/773_14560.html', 'rt', encoding='shift_jis') as f:
        soup = BeautifulSoup(f, 'html.parser')

    body = soup.find('body')

    with open('data/kokoro_transcript.txt', 'wt') as f:
        _process_soup(body)

def ctc_best_path(logits, labels, beam_size=50):
    # Expand label with blanks
    import numpy as np
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
        # Keep the current position
        path1 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score1 = np.zeros([labels.shape[0]], dtype=np.float32)
        k = np.nonzero(alignments[-1] < labels.shape[0])
        v = alignments[-1][k]
        path1[v] = k
        score1[v] = score + logits[i, labels[v]]

        # Move forward
        path2 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score2 = np.zeros([labels.shape[0]], dtype=np.float32)
        k = np.nonzero(alignments[-1] + 1 < labels.shape[0])
        v = (alignments[-1] + 1)[k]
        path2[v] = k
        score2[v] = score + logits[i, labels[v]]

        # Skip blank and move forward
        path3 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score3 = np.zeros([labels.shape[0]], dtype=np.float32)
        k = np.nonzero(alignments[-1] + 2 < labels.shape[0])
        v = (alignments[-1] + 2)[k]
        path3[v] = k
        score3[v] = score + logits[i, labels[v]]

        score = np.where(score1 > score2, score1, score2)
        path = np.where(score1 > score2, path1, path2)
        score = np.where((score3 > score) & (labels != 0), score3, score)
        path = np.where((score3 > score) & (labels != 0), path3, path)

        alignment, = np.nonzero(path >= 0)
        if len(alignment) > beam_size * 2:
            k = np.argsort(score[alignment])
            alignment = alignment[k[-beam_size:]]
            print(len(alignment))
        paths.append(path[alignment])
        score = score[alignment]
        alignments.append(alignment)
        
        if i % 400 == 0:
            s = []
            k = np.arange(alignments[-1].shape[0])
            for path, alighment in zip(reversed(paths), reversed(alignments)):
                #print(path[k], alighment[k])
                s.append(alighment[k])
                k = path[k]
            s = np.array(list(reversed(s))).T
            s = sorted(zip(s, score), key=lambda x: x[1], reverse=True)

            #s = list(reversed(list(zip(s, score))))
            for j, k in s[:3]:
                s = decode_text([labels[x] for x in j])
                print(f'step: {i:4d} {k:.5f} {s}')
        #print('alignment:', alignment.tolist())
        #print('score:', score.tolist())
    return [0], [0]# score, path

def test(model_dir='./model/ctc-20210313'):
    from voice100.train_ctc import Voice100CTCTask
    from voice100.data_pipeline import NORMPARAMS
    import tensorflow as tf
    import re
    from glob import glob
    task = Voice100CTCTask(args)
    model = task.create_model()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=5)
    if not ckpt_manager.latest_checkpoint:
        raise ValueError()
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    a = [
        (file, int(re.sub('.*16000_([0-9]+)_[0-9]+.npz', r'\1', file)))
        for file in glob('data/cache/kokoro/kokoro_001_natsume_64kb_16000_*.npz')
        ]
    a = sorted(a, key=lambda x: x[1])
    a = [file for file, _ in a]
    r = []
    for file in a:
        with np.load(file) as f:
            audio = f['audio']
            r.append(audio)
    audio = np.concatenate(r, axis=0)
    audio = audio[np.newaxis, :, :].astype(np.float32)
    audio = (audio - NORMPARAMS[:, 0]) / NORMPARAMS[:, 1]
    audio_mask = np.ones_like(audio[:, :, 0], dtype=bool)
    audio_len = [audio.shape[1]]
    audio_mask = tf.sequence_mask(audio_len, maxlen=tf.shape(audio)[1])
    logits = model(audio, mask=audio_mask)
    np.savez('a.npz', logits=logits)

def test2():
    with np.load('a.npz') as f:
        logits = f['logits']

    print(logits.shape)
    k = logits[0].argmax(axis=-1)
    print(k.shape)
    k = decode_text(k)
    #print(k)
    #logits = np.concatenate([logits]*10, axis=1)
    s = ''
    with open('data/kokoro_transcript.txt') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            s += parts[1]
    s = s.replace(' ', '')
    labels = encode_text(s)
    print(labels.shape)
    _, p = ctc_best_path(logits[0], labels)
    l = decode_text(p)
    print(l)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--normalize', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', help='Dataset to process, css10ja, tsukuyomi_normal')

    args = parser.parse_args()

    test2()

if __name__ == '__main__':
    main()