from voice100.preprocess import wav2feature, readwav
from glob import glob
import math
import os
import numpy as np
from tqdm import tqdm

pitchshift = math.exp(5.783612067835965 - 4.830453997458316)

files = glob('data/japanese-single-speaker-speech-dataset/meian/*.wav')
for file in tqdm(files):
    fs = 16000
    x = readwav(file, fs)
    feature = wav2feature(x, fs, normed=True, pitchshift=pitchshift)
    outfile = 'tmp/' + os.path.basename(file).replace('.wav', '.npz')
    np.savez(outfile, feature)