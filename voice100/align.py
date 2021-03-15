# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from glob import glob

from .vocoder import encode_audio

def split_voiced(x, silent_threshold, minimum_silent_frames, padding_frames, window_size):
    assert(2 * padding_frames < minimum_silent_frames)
    
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

class IndexDataArray:
    def __init__(self, file):
        self.file = file
        self.current = 0
        self.index = []
        self.data = []

    def append(self, data):
        self.current += data.shape[0]
        self.index.append(self.current)
        self.data.append(data)

    def finish(self):
        np.savez(file,
            index=np.concatenate(self.index, axis=0),
            data=np.concatenate(self.data, axis=0))

def test(filespec):
    window_size = 512 # 46ms
    minimum_silent_duration = 0.5
    padding_duration = 0.05
    minimum_silent_frames = minimum_silent_duration * sr / window_size
    padding_frames = min(1, int(padding_duration * sr // window_size))

    f0_floor, f0_ceil = (57.46701428196299, 196.7528135117272)

    audio_array = IndexDataArray('a.npz')
    for file in sorted(glob(filespec))[:3]:
        x, sr = librosa.load(file)
        for s, e in split_voiced(x, silent_threshold, minimum_silent_frames, padding_frames, window_size) * window_size:
            y = x[s * window_size: e * window_size]
            audio = encode_audio(y, f0_floor, f0_ceil)
            audio_array.append(audio)
    audio_array.finish()
