# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torchaudio

from .encoder import encode_text
from .preprocess import open_index_data_for_write

import logging
logging.basicConfig(level=logging.INFO)

TEXT_PATH = 'data/%s-text.npz'
AUDIO_PATH = 'data/%s-audio.npz'


def readcorpus_css10ja(data_dir):
    from ._css10ja2voca import css10ja2voca
    metafile = os.path.join(data_dir, 'transcript.txt')
    corpus = []
    with open(metafile) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            id_, _, yomi, _ = parts
            assert '..' not in id_  # Just make sure it is under the directory.
            clipfile = os.path.join(data_dir, id_)
            monophone = css10ja2voca(yomi)
            corpus.append((clipfile, monophone))
    return corpus


def readcorpus_kokoro(data_dir, format):
    metafile = os.path.join(data_dir, 'metadata.csv')
    corpus = []
    with open(metafile) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            id_, _, yomi = parts
            assert '..' not in id_  # Just make sure it is under the directory.
            clipfile = os.path.join(data_dir, 'wavs', id_ + '.' + format)
            corpus.append((clipfile, yomi))
    return corpus


def prepare_dataset(args, expected_sample_rate=22050, n_mfcc=40, n_mels=40, n_fft=512):

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=expected_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': n_fft // 2})

    if args.dataset == 'css10ja':
        corpus = readcorpus_css10ja(args.data)
    elif args.dataset == 'kokoro':
        corpus = readcorpus_kokoro(args.data, args.format)
    else:
        raise ValueError()

    with open_index_data_for_write(TEXT_PATH % (args.dataset,)) as textf:
        with open_index_data_for_write(AUDIO_PATH % (args.dataset,)) as audiof:
            for clipfile, monophone in tqdm(corpus):

                if not monophone:
                    print('Skipping: <empty>')
                    continue
                try:
                    encoded = encode_text(monophone)
                    assert encoded.dtype == np.int8
                except:
                    print(f'Skipping: {monophone}')
                    continue
                encoded = encode_text(monophone)

                y, sr = torchaudio.load(clipfile)
                assert len(y.shape) == 2 and y.shape[0] == 1
                assert sr == expected_sample_rate
                y = torch.mean(y, axis=0)  # to mono
                mfcc = mfcc_transform(y).T
                textf.write(encoded)
                audiof.write(mfcc.numpy().astype(np.float32))


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Directory of input speech data')
    parser.add_argument('--dataset', default='kokoro', help='Dataset name')
    parser.add_argument('--format', default='flac', choices=['wav', 'mp3', 'flac'], help='Clip format')
    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == '__main__':
    main_cli()
