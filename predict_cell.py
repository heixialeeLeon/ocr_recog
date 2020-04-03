import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.dataset_cell import *
from models.cell_rcnn import Cell_CRNN, init_weights
from models.simple_crnn import Simple_CRNN
from models.vgg16_crnn import VGG16_CRNN
from utils.alphabets import Alphabets
from utils.img_show import *
from utils.tensor_utils import *
from utils.train_utils import *
import config.config_cell as config

class Predictor(object):
    def __init__(self, model_path, device='cuda'):
        self.alphabets_set = config.alphabets
        self.alphabets = Alphabets(self.alphabets_set)
        self.device = device
        self.input_size = (320,32)
        self.net = VGG16_CRNN(len(self.alphabets_set))
        self.net.load_state_dict(torch.load(model_path))
        self.tensor_process = TensorProcess(self.alphabets)
        self.transform = default_input_transform
        self.net.to(device)
        self.net.eval()

    def __pre_process(self,img):
        img = cv2.resize(img, self.input_size)
        img = Image.fromarray(img, mode="L")
        tensor = self.transform(img)
        tensor = torch.unsqueeze(tensor, dim=0)
        tensor = Variable(tensor).to(self.device)
        return tensor

    def predict(self, img):
        input = self.__pre_process(img)
        output = self.net(input)
        predict_result = self.tensor_process.post_process(output)
        return predict_result