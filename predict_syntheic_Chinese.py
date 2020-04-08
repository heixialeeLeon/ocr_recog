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
from datasets.dataset_syntheic_Chinese import *
from models.vgg16_crnn import VGG16_CRNN, init_weights
from utils.alphabets import *
from utils.tensor_utils import *
import config.config_syntheic_Chinese as config
import random
import numpy as np

net =VGG16_CRNN(len(config.alphabets))

alphabets = Alphabets(config.alphabets)
tensor_process = TensorProcess(alphabets)

device="cuda"

input_transform = Compose([
	ToTensor(),
	# Normalize([.485, .456, .406], [.229, .224, .225]),
])

def resume_model(model, model_path):
    print("Resume model from {}".format(config.resume_model))
    model.load_state_dict(torch.load(model_path))

def predict(net, image_file):
    net.eval()
    net.to(device)
    image = cv2.imread(image_file, 0)
    image = cv2.resize(image, config.image_input_size)
    input = input_transform(image)
    input = torch.unsqueeze(input,dim=0)
    input = Variable(input).to(device)
    output = net(input)
    predict = tensor_process.post_process(output)
    #print("predict: {}".format(predict))
    return image, predict

if __name__ == "__main__":
    from pathlib import Path
    resume_model(net,"checkpoint/recog_chinese.pth")
    image_files = Path('/data/Syntheic_Chinese/images').rglob("*.jpg")
    for item in image_files:
        image, result = predict(net,str(item))
        print("predict: {}".format(result))
        cv2.imshow("image", image)
        cv2.waitKey(0)
