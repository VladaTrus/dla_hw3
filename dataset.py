import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchaudio
from torchvision import transforms
from torch import distributions

mel_spectrogramer = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=256,
    hop_length=128,
    f_min=0,
    f_max=8000,
    n_mels=40,
)

class Squeeze(object):
    def __call__(self, samples):
        return samples.squeeze()

class GaussianNoise(object):
    def __call__(self, wav):
        noiser = distributions.Normal(0, 0.05)
        wav = wav + noiser.sample(wav.size())
        return wav.clamp_(-1, 1)
        

class MelSpectrogram(object):
    def __call__(self, wav):
        mel_spectrogram = mel_spectrogramer(wav)
        return mel_spectrogram

transform_train = transforms.Compose([
        Squeeze(),
        GaussianNoise(),
        MelSpectrogram(),
        torchaudio.transforms.TimeMasking(1, True),
    ])
transform_test = transforms.Compose([
        Squeeze(),
        MelSpectrogram(),
    ])

def make_train_test(dir = 'speech_commands', keyword = 'sheila'):
    all_words = [d for d in os.listdir(dir) if 
                os.path.isdir(os.path.join(dir, d)) and not d.startswith('_')]
    data = []
    for c in all_words:
        d = os.path.join(dir, c)
        target = c
        for f in os.listdir(d):
            path = os.path.join(d, f)
            data.append((path, target))
    df = pd.DataFrame(data, columns=['path', 'word'])
    pos_df = df[df['word'] == keyword]
    for i in range(20):
        df = pd.concat([df, pos_df])
    train_x, test_x, train_y, test_y = train_test_split(df['path'], df['word'], test_size=0.3, random_state=42)
    return SpeechCommandsDataset(train_x, train_y, transform_type ='train'), \
           SpeechCommandsDataset(test_x, test_y, transform_type = 'test')



class SpeechCommandsDataset(Dataset):
    def __init__(self, paths, words, keyword = 'sheila', transform_type = 'train', dir = 'speech_commands'):
        self.paths = paths.values
        self.words = words.values
        self.keyword = keyword
        self.transform = transform_type

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.paths[index])
        pad_wav = torch.full((16000, ), fill_value=0.0)
        if wav.shape[1] < 16000:
            pad_wav[:wav.shape[1]] = wav[:wav.shape[1]]
        else:
            pad_wav[:16000] = wav[:16000]
        if self.transform == 'train':
          item = transform_train(pad_wav).T
        if self.transform == 'test':
          item = transform_test(pad_wav).T
      
        return item, int(self.words[index] == self.keyword)
