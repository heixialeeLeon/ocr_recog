import torch
from pathlib import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch.nn.utils.rnn as rnn
from utils.alphabets import Alphabets
import math

class PAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, tensor):
        c, h, w = tensor.shape
        pad_tensor = torch.FloatTensor(*self.max_size).fill_(0)
        pad_tensor[:, :, :w] =  tensor # right pad
        return pad_tensor

if __name__ == "__main__":
    input =  torch.zeros(1, 32, 320)
    pad = PAD((1,32,500))
    output = pad(input)
    print(output.shape)