# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import numpy as np
import math
from tqdm import tqdm
import argparse

from .vocoder import readaudio, readwav, estimatef0, encode_audio
from .encoder import encode_text

import logging
logging.basicConfig(level=logging.INFO)

CORPUSDATA_PATH = 'data/balance_sentences.txt'
CORPUSDATA_CSS10JA_PATH = 'data/japanese-single-speaker-speech-dataset/transcript.txt'

WAVDATA_PATH = {
    'css10ja': 'data/japanese-single-speaker-speech-dataset/%s',
    'tsuchiya_normal': 'data/tsuchiya_normal/tsuchiya_normal_%s.wav',
    'hiroshiba_normal': 'data/hiroshiba_normal/hiroshiba_normal_%s.wav',
    'tsukuyomi_normal': 'data/つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠）'
        '/01 WAV（収録時の音量のまま）/VOICEACTRESS100_%s.wav',
}

# F0 mean +/- 2.5 std
F0_RANGE = {
    'css10ja': (57.46701428196299, 196.7528135117272),
    'tsukuyomi_normal': (138.7640311667663, 521.2003965068923)
}

TEXT_PATH = 'data/%s_text.npz'
AUDIO_PATH = 'data/%s_audio_%d.npz'

def readcorpus(file):
    corpus = []
    with open(file) as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            id_, _, _, monophone = parts
            monophone = monophone.replace('/', '').replace(',', '')
            corpus.append((id_, monophone))

    return corpus

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

def split_voiced(x, minimum_silent_frames, padding_frames, window_size):
    assert 2 * padding_frames < minimum_silent_frames
    
    num_frames = len(x) // window_size
    mX = np.mean(x[:window_size * num_frames].reshape((-1, window_size)) ** 2, axis=1)
    mX = 10 * np.log(mX)

    silent_threshold = (np.max(mX) + np.min(mX)) / 2
    
    voiced = mX > silent_threshold

    silent_to_voiced = np.where((~voiced[:-1]) & voiced[1:])[0] + 1 # The position where the voice starts
    voiced_to_silent = np.where((voiced[:-1]) & ~voiced[1:])[0] + 1 # The position where the silence starts
    if not voiced[0]:
        # Eliminate the preceding silence
        silent_to_voiced = silent_to_voiced[1:]
    if not voiced[-1]:
        # Eliminate the succeeding silence
        voiced_to_silent = voiced_to_silent[:-1]
    silent_ranges = np.stack([voiced_to_silent, silent_to_voiced]).T

    for s, e in silent_ranges:
        if e - s < minimum_silent_frames:
            voiced[s:e] = True

    silent_to_voiced = np.where((~voiced[:-1]) & voiced[1:])[0] + 1 # The position where the voice starts
    voiced_to_silent = np.where((voiced[:-1]) & ~voiced[1:])[0] + 1 # The position where the silence starts
    if voiced[0]:
        # Include the preceding voiced
        silent_to_voiced = np.insert(silent_to_voiced, 0, 0)
    if voiced[-1]:
        # Include the succeeding voiced
        voiced_to_silent = np.append(voiced_to_silent, len(voiced))
    voiced_ranges = np.stack([silent_to_voiced, voiced_to_silent]).T
    
    return voiced_ranges + np.array([[-padding_frames, padding_frames]])

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
                for start, end in split_voiced(x, minimum_silent_frames, padding_frames, window_size) * window_size:
                    y = x[start:end].astype(np.float64)
                    audioname = os.path.basename(file)
                    cache_file = 'data/cache/%s/%s.%d_%08d_%08d.npz' % (args.dataset, audioname, sr, start, end)
                    if os.path.exists(cache_file):
                        with np.load(cache_file) as f:
                            audio = f['audio']
                    else:
                        audio = encode_audio(y, f0_floor, f0_ceil)
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                        np.savez(cache_file, audio=audio)
                    data.write(audio.astype(np.float32))
                    segf.write(f'{file}|{start}|{end}\n')

def analyze_files(name, files, eps=1e-20):
    f0_list = []
    power_list = []
    for file in tqdm(files):
        x = readwav(file)

        # Estimate F0
        f0 = estimatef0(x)
        f0 = f0[f0 > 0].copy()
        assert f0.ndim == 1
        f0_list.append(f0)

        # Calculate loudness
        window_size = 1024
        power = x[:(x.shape[0] // window_size) * window_size].reshape((-1, window_size))
        power = np.log(np.mean(power ** 2, axis=1) + eps)
        assert power.ndim == 1
        power_list.append(power)

    f0 = np.concatenate(f0_list, axis=0)
    f0_mean = np.mean(f0)
    f0_std = np.std(f0)
    f0_hist, f0_bin_edges = np.histogram(f0, bins=np.linspace(0, 1000.0, 101), density=True)

    power = np.concatenate(power_list, axis=0)
    power_mean = np.mean(power)
    power_std = np.std(power)
    power_hist, power_bin_edges = np.histogram(power, bins=np.linspace(-30, 0.0, 100), density=True)

    np.savez(os.path.join('data', '%s_stat.npz' % name),
        f0=f0,
        f0_mean=f0_mean, f0_std=f0_std,
        f0_hist=f0_hist, f0_bin_edges=f0_bin_edges,
        power_mean=power_mean, power_std=power_std,
        power_hist=power_hist, power_bin_edges=power_bin_edges)

def analyze_css10ja(name):
    corpus = readcorpus_css10ja(CORPUSDATA_CSS10JA_PATH)
    files = []
    for id_, monophone in corpus[17::6841 // 100]: # Use 100 samples from corpus
        file = os.path.join('data', 'japanese-single-speaker-speech-dataset', id_)
        files.append(file)
    analyze_files(name, files)

def analyze_jvs(name):
    corpus = readcorpus(CORPUSDATA_PATH)
    files = []
    for id_, monophone in corpus:
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        files.append(file)
    analyze_files(name, files)

def preprocess_css10ja(name):

    if name.endswith('_highpitch'):
        tsukuyomi_average_logf0 = 5.783612067835965
        css10ja_average_logf0 = 4.830453997458316
        pitchshift = math.exp(tsukuyomi_average_logf0 - css10ja_average_logf0)
    else:
        pitchshift = None
    f0_floor, f0_ceil = F0_RANGE[name]

    text_array = IndexDataArray(TEXT_PATH % (name,))
    audio_array = IndexDataArray(AUDIO_PATH % (name, 16000))

    corpus = readcorpus_css10ja(CORPUSDATA_CSS10JA_PATH)
    for id_, monophone in tqdm(corpus[:10]):

        if not monophone:
            print('Skipping: <empty>')
            continue
        try:
            text = encode_text(monophone)
        except:
            print(f'Skipping: {monophone}')
            continue
    
        file = os.path.join('data', 'japanese-single-speaker-speech-dataset', id_)
        assert '..' not in file # Just make sure it is under the current directory.
        cache_file = os.path.join('data', 'cache', name, id_.replace('.wav', '.npz'))
        if os.path.exists(cache_file):
            audio = np.load(cache_file)['audio']
            assert audio.shape[0] > 0
        else:
            x = readwav(file)
            audio = encode_audio(x, f0_floor, f0_ceil, pitchshift=pitchshift)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez(cache_file, audio=audio)

        text_array.append(text.astype(np.int8))
        audio_array.append(audio.astype(np.float32))

    text_array.finish()
    audio_array.finish()

def preprocess_jvs(name):
    corpus = readcorpus(CORPUSDATA_PATH)
    f0_floor, f0_ceil = F0_RANGE[name]
    data = make_empty_data()
    text_index = 0
    audio_index = 0
    for id_, monophone in tqdm(corpus):
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        x = readwav(file)
        text = encode_text(monophone)
        audio = encode_audio(x, f0_floor, f0_ceil)
        text_index += text.shape[0]
        audio_index += audio.shape[0]
        append_data(data, id_, text_index, text, audio_index, audio)

    finish_data(data, OUTPUT_PATH % (name, "train"))

def normalize_css10ja(name):

    if False:
        from .data_pipeline import normparams
        for split in 'train', 'val':
            with np.load(OUTPUT_PATH % (name, split) + '.bak') as f:
                data = {k:v for k, v in f.items()}
            print(data.keys())
            print(data['audio_data'].shape)
            data['audio_data'] = data['audio_data'] * normparams[:, 0] + normparams[:, 1]
            np.savez(OUTPUT_PATH % (name, split), **data)

    with np.load(OUTPUT_PATH % (name, "train")) as f:
        audio = f['audio_data']
    mean = np.mean(audio, axis=0)
    std = np.std(audio, axis=0)
    x = np.stack([mean, std], axis=1)
    for i in range(x.shape[0]):
        print('    [%s, %s],' % (str(x[i, 0]), str(x[i, 1])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', action='store_true', help='Split audio and encode with WORLD vocoder.')
    parser.add_argument('--analyze', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--normalize', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', required=True, help='Dataset to process, css10ja, tsukuyomi_normal')
    args = parser.parse_args()

    if args.split:
        split_audio(args)
    elif args.analyze:
        if args.dataset == 'css10ja':
            analyze_css10ja(args.dataset)
        else:
            analyze_jvs(args.dataset)
    elif args.normalize:
        if args.dataset == 'css10ja':
            normalize_css10ja(args.dataset)
        else:
            assert False
    else:
        if args.dataset == 'css10ja' or args.dataset == 'css10ja_highpitch':
            preprocess_css10ja(args.dataset)
        else:
            preprocess_jvs(args.dataset)
