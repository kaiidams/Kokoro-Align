import soundfile as sf
import os
import numpy as np
from tqdm import tqdm
from .vocoder import writewav, decode_audio
from .encoder import decode_text

def test_data(name='css10ja'):
    file = 'data/%s_train.npz' % name
    f = np.load(file)
    data = {k: v for k, v in f.items()}
    for index in tqdm(range(10)):
        text_start = data['text_index'][index - 1] if index else 0
        text_end = data['text_index'][index]
        audio_start = data['audio_index'][index - 1] if index else 0
        audio_end = data['audio_index'][index]
        text = data['text_data'][text_start:text_end]
        audio = data['audio_data'][audio_start:audio_end, :]
        print(decode_text(text))
        x = decode_audio(audio)
        file = 'data/synthesized/%s_%04d.wav' % (name, index)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writewav(file, x)

def main():
    test_data('tsukuyomi_normal')

if __name__ == '__main__':
    main()