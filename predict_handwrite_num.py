import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from datasets.dataset_syntheic_Chinese import *
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from models.vgg16_crnn import VGG16_CRNN, init_weights
from models.resnet_crnn import Resnet_CRNN
from utils.alphabets import *
from utils.tensor_utils import *
from datasets.tools import PAD
import config.config_handwrite_nums as config
import random
import numpy as np
import cv2

#net =VGG16_CRNN(len(config.alphabets))
net = Resnet_CRNN(len(config.alphabets),1)

alphabets = Alphabets_Chinese(config.alphabets)
tensor_process = TensorProcess(alphabets)

device="cuda"

input_transform = Compose([
	ToTensor(),
	# Normalize([.485, .456, .406], [.229, .224, .225]),
])

def resume_model(model, model_path):
    print("Resume model from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))

def predict(net, image_file):
    pad = PAD((1, config.image_input_size[1], config.image_input_size[0]))
    net.eval()
    net.to(device)
    image = cv2.imread(image_file, 0)
    h,w = image.shape
    scale = 32/h
    image = cv2.resize(image,None,fx=scale, fy=scale)
    #image = cv2.resize(image, config.image_input_size)
    input = input_transform(image)
    #input = pad(input)
    input = torch.unsqueeze(input,dim=0)
    input = Variable(input).to(device)
    output = net(input)
    predict = tensor_process.post_process_ch(output)
    #print("predict: {}".format(predict))
    return image, predict

if __name__ == "__main__":
    from pathlib import Path
    #resume_model(net,"checkpoint_ch/cn_v0.pth")
    #resume_model(net,"checkpoint_ch/vgg16_v0.pth")
    #resume_model(net, "checkpoint_handwrite/resnet.pth")
    resume_model(net, "checkpoint/recog_epoch_20.pth")
    image_files = Path('/home/peizhao/data/ocr/handwrite/real_data').glob("*.png")
    for item in image_files:
        image, result = predict(net,str(item))
        print("predict: {}, file:{}".format(result, item))
        cv2.imshow("image", image)
        cv2.waitKey(0)
