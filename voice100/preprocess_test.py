import soundfile as sf
import os
import numpy as np
import argparse
from tqdm import tqdm
from .vocoder import writewav, decode_audio
from .encoder import decode_text

def test_text_data():
    file = 'data/%s_text.npz' % args.dataset
    with np.load(file) as f:
        text_index = f['indices']
        text_data = f['data']
    file = 'data/%s_audio_16000.npz' % args.dataset
    with np.load(file) as f:
        audio_index = f['indices']
        audio_data = f['data']

    for index in tqdm(range(10)):
        text_start = text_index[index - 1] if index else 0
        text_end = text_index[index]
        audio_start = audio_index[index - 1] if index else 0
        audio_end = audio_index[index]
        text = text_data[text_start:text_end]
        audio = audio_data[audio_start:audio_end, :]
        print(decode_text(text))
        x = decode_audio(audio)
        file = 'data/synthesized/%s_%04d.wav' % (args.dataset, index)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writewav(file, x)

def test_audio_data(args):
    file = 'data/%s_audio_16000.npz' % args.dataset
    with np.load(file) as f:
        audio_index = f['indices']
        audio_data = f['data']

    for index in tqdm(range(10)):
        audio_start = audio_index[index - 1] if index else 0
        audio_end = audio_index[index]
        audio = audio_data[audio_start:audio_end, :]
        x = decode_audio(audio)
        file = 'data/synthesized/%s_%04d.wav' % (args.dataset, index)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writewav(file, x)

def test_audio_split(args):
    from .vocoder import readaudio, writewav
    x = readaudio('data/gongitsune_um_librivox_64kb_mp3/gongitsune_01_niimi_64kb.mp3')
    writewav('a.wav', x[371200:403456])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='Analyze F0 of sampled data.')
    parser.add_argument('--normalize', action='store_true', help='Compute normalization parameters.')
    parser.add_argument('--dataset', help='Dataset to process, css10ja, tsukuyomi_normal')

    args = parser.parse_args()
    #test_audio_data(args)
    test_audio_split(args)

if __name__ == '__main__':
    main()