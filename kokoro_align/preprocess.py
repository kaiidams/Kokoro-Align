# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import argparse
import torch
import torchaudio

import logging
logging.basicConfig(level=logging.INFO)


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
    silent_to_voiced = np.where((~voiced[:-1]) & voiced[1:])[0] + 1  # The position where the voice starts
    voiced_to_silent = np.where((voiced[:-1]) & ~voiced[1:])[0] + 1  # The position where the silence starts
    if not voiced[0]:
        # Eliminate the preceding silence
        silent_to_voiced = silent_to_voiced[1:]
    if not voiced[-1]:
        # Eliminate the succeeding silence
        voiced_to_silent = voiced_to_silent[:-1]
    return np.stack([voiced_to_silent, silent_to_voiced]).T


def get_split_points(x, minimum_silent_frames, minimum_split_distance, maximum_split_distance, window_size, eps=1e-12):

    num_frames = len(x) // window_size
    mX = np.mean(x[:window_size * num_frames].reshape((-1, window_size)) ** 2, axis=1)
    mX = 10 * np.log(mX + eps)

    silent_threshold = (np.max(mX) + np.min(mX)) / 2

    while True:
        # Fill short silent
        voiced = mX > silent_threshold
        silent_ranges = get_silent_ranges(voiced)
        for s, e in silent_ranges:
            if e - s < minimum_silent_frames:
                voiced[s:e] = True

        silent_ranges = get_silent_ranges(voiced)

        # Split in the center of silence.
        silent_points = (silent_ranges[:, 0] + silent_ranges[:, 1]) // 2

        split_distance = (
            np.append(silent_points, num_frames) - np.insert(silent_points, 0, 0))
        if np.max(split_distance) < maximum_split_distance:
            break

        minimum_silent_frames *= 0.5
        if minimum_silent_frames < 0.05:
            raise ValueError("Audio cannot be split into")

    # Merge short splits
    while len(silent_points):
        split_distance = (
            np.append(silent_points, num_frames) - np.insert(silent_points, 0, 0))
        i = np.argmin(split_distance)
        if split_distance[i] > minimum_split_distance:
            break
        if i == 0:
            silent_points = np.delete(silent_points, i)
        elif i == len(silent_points):
            silent_points = np.delete(silent_points, len(silent_points) - 1)
        else:
            if split_distance[i - 1] < split_distance[i + 1]:
                silent_points = np.delete(silent_points, i - 1)
            else:
                silent_points = np.delete(silent_points, i)

    return silent_points


def split_audio(
    audio_file, segment_file, audio_data_file,
    expected_sample_rate=22050, n_mfcc=40, n_mels=40, n_fft=512
):
    window_size = n_fft // 2  # 46ms
    minimum_silent_duration = 0.25  # 500ms
    # padding_duration = 0.05  # 50ms
    minimum_silent_frames = minimum_silent_duration * expected_sample_rate / window_size
    minimum_split_distance = 3.0 * expected_sample_rate / window_size
    maximum_split_distance = 15.0 * expected_sample_rate / window_size
    # padding_frames = min(1, int(padding_duration * expected_sample_rate // window_size))

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=expected_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': n_fft // 2})

    with open(segment_file, 'wt') as segf:
        with open_index_data_for_write(audio_data_file) as data:
            y, sr = torchaudio.load(audio_file)
            assert len(y.shape) == 2 and y.shape[0] == 1
            assert sr == expected_sample_rate
            y = torch.mean(y, axis=0)  # to mono
            split_points = get_split_points(
                y.numpy(), minimum_silent_frames,
                minimum_split_distance, maximum_split_distance, window_size) * window_size
            for i in range(len(split_points) + 1):
                start = split_points[i - 1] if i > 0 else 0
                end = split_points[i] if i < len(split_points) else len(y)
                mfcc = mfcc_transform(y[start:end]).T
                data.write(mfcc.numpy().astype(np.float32))
                segf.write(f'{end}\n')


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=str, help='Input file')
    parser.add_argument('segment', type=str, help='Segment file')
    parser.add_argument('mfcc', type=str, help='MFCC file')
    args = parser.parse_args()
    split_audio(args.audio, args.segment, args.mfcc)


if __name__ == '__main__':
    main_cli()
