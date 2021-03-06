import torch
import numpy as np

def _readdata(file):
    f = np.load(file)
    return {k:v for k, v in f.items()}

class Voice100Dataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.data = _readdata(file)

    def __getitem__(self, index):
        id_ = self.data['id'][index]
        text_start = self.data['text_index'][index - 1] if index else 0
        text_end = self.data['text_index'][index]
        audio_start = self.data['audio_index'][index - 1] if index else 0
        audio_end = self.data['audio_index'][index]
        text = self.data['text_data'][text_start:text_end]
        audio = self.data['audio_data'][audio_start:audio_end, :]
        assert text_start < text_end
        assert audio_start < audio_end
        return id_, text, audio

    def __len__(self):
        return len(self.data['id'])

    def xxxx(self):
        scale = np.vstack([
            train_data['audio_data'].max(axis=0)
            -train_data['audio_data'].min(axis=0)
        ]).max(axis=0)
        np.clip(scale, 1.05, 1000.0, scale)
        #train_data['audio_data'] *= 1 / scale
        #val_data['audio_data'] *= 1 / scale
