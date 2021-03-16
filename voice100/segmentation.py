# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import numpy as np
from tqdm import tqdm
from voice100.encoder import encode_text, decode_text

class Voice100Dataset:
    def __init__(self, file):
        with np.load(file) as f:
            self.audio_indices = f['indices']
            self.audio_data = f['data']
            #self.audio_indices = f['audio_index']
            #self.audio_data = f['audio_data']

    def __len__(self):
        return len(self.audio_indices)

    def __getitem__(self, index):
        audio_start = self.audio_indices[index - 1] if index else 0
        audio_end = self.audio_indices[index]
        audio = self.audio_data[audio_start:audio_end, :]
        return audio

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

    def get_path(paths, alignments, score):
        s = []
        k = np.arange(alignments[-1].shape[0])
        for path, alighment in zip(reversed(paths), reversed(alignments)):
            s.append(alighment[k])
            k = path[k]
        s = np.array(list(reversed(s))).T # (beam_size, audio_seq_len)
        si = np.argsort(score)[::-1]
        return s[si], score[si]

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
        paths.append(path[alignment])
        score = score[alignment]
        alignments.append(alignment)
        
        if i % 400 == 0:
            best_path, score = get_path(paths, alignments, score)
            for p, s in zip(best_path, score):
                t = decode_text([labels[a] for a in p])
                print(f'step: {i:4d} {s:.5f} {t}')
                #print('alignment:', alignment.tolist())
                #print('score:', score.tolist())

    return get_path(paths, alignments, score)

def test(args, model_dir='./model/ctc-20210313'):
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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, task.params['audio_dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    ])
    def eval_step(audio, audio_len):
        audio_mask = tf.sequence_mask(audio_len, maxlen=tf.shape(audio)[1])
        logits = model(audio, mask=audio_mask)
        logits = tf.nn.softmax(logits, axis=2)
        return logits

    from .preprocess import open_index_data_for_write

    ds = Voice100Dataset('data/%s_audio_16000.npz' % args.dataset)
    #ds = Voice100Dataset('data/%s_train.npz' % args.dataset)
    with open_index_data_for_write('data/%s_phoneme_16000.npz' % args.dataset) as file:
        for example in tqdm(ds):
            audio = example
            audio = (audio - NORMPARAMS[:, 0]) / NORMPARAMS[:, 1]
            audio = audio[np.newaxis, :, :]
            audio_len = [audio.shape[1]]
            logits = eval_step(audio, audio_len)
            x = np.argmax(logits.numpy(), axis=2)
            #print(decode_text(x[0]))
            file.write(logits[0])

def test2(args):
    with np.load('data/%s_phoneme_16000.npz' % args.dataset) as f:
        logits = f['data']
    #with np.load('a.npz') as f:
    #    logits = f['logits']

    print(logits.shape)
    if False:
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
    best_path, score = ctc_best_path(logits[0], labels)
    np.savez('data/kokoro_best_path.npz', best_path=best_path[0], score=score[0])
    l = decode_text([0 if x % 2 == 0 else labels[x // 2] for x in best_path[0]])
    print(l)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--normalize', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', help='Dataset to process, css10ja, tsukuyomi_normal')

    args = parser.parse_args()

    test2(args)

if __name__ == '__main__':
    main()