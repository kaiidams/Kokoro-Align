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
            #self.audio_indices = f['audio_index']
            #self.audio_data = f['audio_data']

    def __len__(self):
        return len(self.audio_indices)

    def __getitem__(self, index):
        audio_start = self.audio_indices[index - 1] if index else 0
        audio_end = self.audio_indices[index]
        audio = self.audio_data[audio_start:audio_end, :]
        return audio

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
        score1[v] = score[k] + logits[i, labels[v]]

        # Move forward
        path2 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score2 = np.zeros([labels.shape[0]], dtype=np.float32)
        k = np.nonzero(alignments[-1] + 1 < labels.shape[0])
        v = (alignments[-1] + 1)[k]
        path2[v] = k
        score2[v] = score[k] + logits[i, labels[v]]

        # Skip blank and move forward
        path3 = np.zeros([labels.shape[0]], dtype=np.int32) - 1
        score3 = np.zeros([labels.shape[0]], dtype=np.float32)
        k = np.nonzero(alignments[-1] + 2 < labels.shape[0])
        v = (alignments[-1] + 2)[k]
        path3[v] = k
        score3[v] = score[k] + logits[i, labels[v]]

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
        
        if False:
            if i % 400 == 0:
                best_path, score = get_path(paths, alignments, score)
                for p, s in zip(best_path, score):
                    t = decode_text([labels[a] for a in p])
                    print(f'step: {i:4d} {s:.5f} {t}')
                    #print('alignment:', alignment.tolist())
                    #print('score:', score.tolist())

    return get_path(paths, alignments, score)

def phoneme(args):
    from voice100.train_ctc import Voice100CTCTask
    from voice100.data_pipeline import NORMPARAMS
    import tensorflow as tf
    import re
    from glob import glob

    sr = 16000
    task = Voice100CTCTask(args)
    model = task.create_model()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=5)
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

    ds = Voice100Dataset('data/%s_audio_%d.npz' % (args.dataset, sr))
    with open_index_data_for_write('data/%s_phoneme_%d.npz' % (args.dataset, sr)) as file:
        with open('data/%s_phoneme_%d.txt' % (args.dataset, sr), 'wt') as txtfile:
            for example in tqdm(ds):
                audio = example
                audio = (audio - NORMPARAMS[:, 0]) / NORMPARAMS[:, 1]
                audio = audio[np.newaxis, :, :]
                audio_len = [audio.shape[1]]
                logits = eval_step(audio, audio_len)
                x = np.argmax(logits.numpy(), axis=2)

                file.write(logits[0])
                txtfile.write(merge_repeated(decode_text(x[0])) + '\n')

def best_path(args):
    sr = 16000
    with np.load('data/%s_phoneme_%d.npz' % (args.dataset, sr)) as f:
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
    with open('data/%s_transcript.txt' % (args.dataset)) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            s += parts[1]
    s = s.replace(' ', '')

    labels = encode_text(s)
    print(labels.shape)
    best_path, score = ctc_best_path(logits, labels)
    np.savez('data/%s_best_path.npz' % (args.dataset), best_path=best_path[0], score=score[0])
    l = decode_text([0 if x % 2 == 0 else labels[x // 2] for x in best_path[0]])
    print(l)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phoneme', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--best_path', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', help='Dataset to process, css10ja, tsukuyomi_normal')
    parser.add_argument('--model_dir', default='model/ctc-20210313')

    args = parser.parse_args()

    if args.phoneme:
        phoneme(args)
    elif args.best_path:
        best_path(args)

if __name__ == '__main__':
    main()