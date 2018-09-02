from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
import os
import torch
import re
import json

from Lang import Lang

MAX_LENGTH = 50
SOS_token = torch.zeros(1947)
SOS_token[0] = 1

EOS_token = torch.zeros(1947)
EOS_token[1] = 1

teacher_forcing_ratio = 0.5
plt.switch_backend('agg')


class VideoDataset(Dataset):
    def __init__(self, path, length, is_train=True):
        self.path=path
        self.is_train=is_train


        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        capLang = Lang(name="EnglishCaptions")
        pair = self.getFeatureVocabPair(self.path, i)
        capLang.addSentence(pair[1])

        training_pair = tensorsFromPair(capLang, pair)

        return training_pair

    def getFeatureVocabPair(self, file, i):
        video = "video" + str(i) + ".npy"
        path = os.path.join(file, "video" + str(i) + ".npy")

        print("Loading: " + video)

        try:
            feature = np.load(path)
            caption = self.getCaption(i)
        except:
            print("Failed to load: " + path)

        return feature, caption, i

    def getCaption(self, id):
        f = open("info.json", 'r' , encoding='utf8')
        info = json.load(f)
        for v in info["sentences"]:
            if(v["video_id"] == "video" + str(id)):
                return self.normalizeString(v["caption"])

    def normalizeString(self, s):
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    #indexes = torch.tensor(indexes, dtype=torch.long)
    final = torch.tensor([])
    for i in indexes:
        if(len(final) == 0):
            final = torch.zeros(lang.n_words)
            final[torch.tensor(i)]
        else:
            temp = torch.zeros(lang.n_words)
            temp[torch.tensor(i ,dtype=torch.long)] = 1
            final = torch.cat((final , temp) , dim=0)
    
    final = torch.cat((final , EOS_token))
    print(final.shape)
    final = final.view(-1 , lang.n_words)
    return final

def tensorsFromPair(input_lang, pair):
    target_tensor = tensorFromSentence(input_lang, pair[1])
    input_tensor = torch.tensor(pair[0] , dtype=torch.float)
    return (input_tensor, target_tensor)


def getGenerator(path, batch_size, length, shuffle=True, train=True):
    ds = VideoDataset(path, length, is_train=train)
    generator = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return generator

if __name__ == '__main__':
    gen = getGenerator('featuresBig', 5, 999)

    for tp in gen:
        print(l)
