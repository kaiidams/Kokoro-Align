# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torchaudio

from .encoder import encode_text

import logging
logging.basicConfig(level=logging.INFO)

CORPUSDATA_CSS10JA_PATH = 'data/japanese-single-speaker-speech-dataset'

TEXT_PATH = 'data/%s_text.npz'
AUDIO_PATH = 'data/%s_audio.npz'

def readcorpus_css10ja(file):
    from ._css10ja2voca import css10ja2voca
    corpus = []
    with open(file) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            id_, _, yomi, _ = parts
            monophone = css10ja2voca(yomi)
            corpus.append((id_, monophone))
    return corpus

class IndexDataArray:
    def __init__(self, file):
        self.file = file
        self.current = 0
        self.indices = []
        self.data = []

    def __enter__(self):
        return self

    def write(self, data):
        self.current += data.shape[0]
        self.indices.append(self.current)
        self.data.append(data)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            indices = np.array(self.indices, dtype=np.int32)
            data = np.concatenate(self.data, axis=0)
            np.savez(self.file, indices=indices, data=data)

def open_index_data_for_write(file):
    return IndexDataArray(file)

def get_silent_ranges(voiced):
    silent_to_voiced = np.where((~voiced[:-1]) & voiced[1:])[0] + 1 # The position where the voice starts
    voiced_to_silent = np.where((voiced[:-1]) & ~voiced[1:])[0] + 1 # The position where the silence starts
    if not voiced[0]:
        # Eliminate the preceding silence
        silent_to_voiced = silent_to_voiced[1:]
    if not voiced[-1]:
        # Eliminate the succeeding silence
        voiced_to_silent = voiced_to_silent[:-1]
    return np.stack([voiced_to_silent, silent_to_voiced]).T

def get_split_points(x, minimum_silent_frames, window_size):
    
    num_frames = len(x) // window_size
    mX = np.mean(x[:window_size * num_frames].reshape((-1, window_size)) ** 2, axis=1)
    mX = 10 * np.log(mX)

    silent_threshold = (np.max(mX) + np.min(mX)) / 2
    
    voiced = mX > silent_threshold
    silent_ranges = get_silent_ranges(voiced)
    for s, e in silent_ranges:
        if e - s < minimum_silent_frames:
            voiced[s:e] = True

    silent_ranges = get_silent_ranges(voiced)

    return (silent_ranges[:, 0] + silent_ranges[:, 1]) // 2

def split_audio(args):
    sr = 16000
    window_size = 512 # 46ms
    minimum_silent_duration = 0.5 # 500ms
    padding_duration = 0.05 # 50ms
    minimum_silent_frames = minimum_silent_duration * sr / window_size
    padding_frames = min(1, int(padding_duration * sr // window_size))
    f0_floor, f0_ceil = (57.46701428196299, 196.7528135117272) # Low male voice

    # Reading audio files
    audio_list_file = f'data/{args.dataset}_audio_files.txt'
    logging.info('Reading list of audio files from %s', audio_list_file)
    with open(audio_list_file) as f:
        audio_files = [line.rstrip('\r\n') for line in f.readlines()]
        audio_files = [file for file in audio_files if file]
    assert all(os.path.exists(file) for file in audio_files)

    audio_segment_file = f'data/{args.dataset}_segment_{sr}.txt'
    audio_data_file = f'data/{args.dataset}_audio_{sr}.npz'

    with open(audio_segment_file, 'w') as segf:
        with open_index_data_for_write(audio_data_file) as data:
            for file in tqdm(audio_files):
                x = readaudio(file, sr)
                split_points = get_split_points(x, minimum_silent_frames, window_size) * window_size
                for i in range(len(split_points) + 1):
                    start = split_points[i - 1] if i > 0 else 0
                    end = split_points[i] if i < len(split_points) else len(x)
                    audioname = os.path.basename(file)
                    cache_file = 'data/cache/%s/%s.%d_%08d_%08d.npz' % (args.dataset, audioname, sr, start, end)
                    if os.path.exists(cache_file):
                        with np.load(cache_file) as f:
                            audio = f['audio']
                    else:
                        y = x[start:end].astype(np.float64)
                        audio = encode_audio(y, f0_floor, f0_ceil)
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                        np.savez(cache_file, audio=audio)
                        data.write(audio.astype(np.float32))
                        segf.write(f'{file}|{start}|{end}\n')

def preprocess_css10ja(args, expected_sample_rate=22050, n_mfcc=40, n_mels=40, n_fft=1024):

    args.dataset = 'css10ja'

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=expected_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': n_fft // 2})

    corpus = readcorpus_css10ja(os.path.join(CORPUSDATA_CSS10JA_PATH, 'transcript.txt'))
    with open_index_data_for_write(TEXT_PATH % (args.dataset,)) as textf:
        with open_index_data_for_write(AUDIO_PATH % (args.dataset,)) as audiof:
            for id_, monophone in tqdm(corpus):

                if not monophone:
                    print('Skipping: <empty>')
                    continue
                try:
                    encoded = encode_text(monophone)
                    assert encoded.dtype == np.int8
                except:
                    print(f'Skipping: {monophone}')
                    continue
            
                file = os.path.join(CORPUSDATA_CSS10JA_PATH, id_)
                assert '..' not in file # Just make sure it is under the current directory.
                y, sr = torchaudio.load(file)
                assert y.shape[0] == 1
                assert sr == expected_sample_rate
                y = y[0]
                mfcc = mfcc_transform(y).T
                textf.write(encoded)
                audiof.write(mfcc.numpy().astype(np.float32))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', action='store_true', help='Split audio and encode with WORLD vocoder.')
    parser.add_argument('--css10ja', action='store_true', help='Analyze F0 of sampled data.')
    args = parser.parse_args()

    if args.split:
        split_audio(args)
    elif args.css10ja:
        preprocess_css10ja(args)
    else:
        raise ValueError('Unknown command')