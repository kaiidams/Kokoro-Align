import soundfile as sf
import os
import numpy as np
from tqdm import tqdm
from .vocoder import writewav, decode_audio
from .encoder import decode_text

def test_data(name='css10ja'):
    file = 'data/%s_text.npz' % name
    with np.load(file) as f:
        text_index = f['index']
        text_data = f['data']
    file = 'data/%s_audio_16000.npz' % name
    with np.load(file) as f:
        audio_index = f['index']
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
        file = 'data/synthesized/%s_%04d.wav' % (name, index)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writewav(file, x)

def main():
    test_data()#'tsukuyomi_normal')

if __name__ == '__main__':
    main()