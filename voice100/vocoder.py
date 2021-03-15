# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import librosa
import numpy as np

SAMPLE_RATE = 22050

def readwav(file, fs=SAMPLE_RATE):
    x, origfs = librosa.load(file)
    if fs is not None:
        x = librosa.resample(x, origfs, fs)
    x = x / x.max()
    return x

def encode_audio(x, sr=SAMPLE_RATE):
    mel_spec = librosa.feature.melspectrogram(x, sr=sr, n_fft=400,
                                            hop_length=200, win_length=400, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
