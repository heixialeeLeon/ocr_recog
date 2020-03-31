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
from dataset.dataset import *
from models.simple_crnn import Simple_CRNN, init_weights
from utils.alphabets import Alphabets
from utils.tensor_utils import *
import configs as config

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# training parameters
# parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--data_dir", type=str, default="/data/captcha/330/train",help="data dir location")
parser.add_argument("--batch_size", type=int, default=32,help="batch size")
parser.add_argument("--epochs", type=int, default=300,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=50,help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--steps_show", type=int, default=100,help="steps per epoch")
parser.add_argument("--scheduler_step", type=int, default=10,help="scheduler_step for epoch")
parser.add_argument("--weight", type=str, default=None,help="weight file for restart")
parser.add_argument("--output_path", type=str, default="checkpoint",help="checkpoint dir")
parser.add_argument("--devices", type=str, default="cuda",help="device description")
parser.add_argument("--image_channels", type=int, default=3,help="batch image_channels")
parser.add_argument("--resume_model", type=str, default='checkpoint/recog_v1.pth', help="resume model path")

args = parser.parse_args()
print(args)

def save_model_as(model, model_name):
    ckpt_name = '/'+model_name
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def save_model(model,epoch):
    '''save model for eval'''
    ckpt_name = '/recog_epoch_{}.pth'.format(epoch)
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def resume_model(model, model_path):
    print("Resume model from {}".format(config.resume_model))
    model.load_state_dict(torch.load(model_path))

def model_to_device(model):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def tensor_to_device(tensor):
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    return tensor.to(device)

# prepare the data
train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate_fn)
test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1, shuffle=False, collate_fn=default_collate_fn)
test_loader_batch = DataLoader(dataset=test_dataset, num_workers=4, batch_size=16, shuffle=True, collate_fn=default_collate_fn)

# prepare the loss
criterion = nn.CTCLoss()

# prepare the network
net =Simple_CRNN(len(config.alphabets),input_channels=3, hidden_unit=128)

# prepare the optim
#optimizer = torch.optim.SGD(net.parameters(),args.lr)
#optimizer = torch.optim.SGD(parameters_settings, args.lr)
optimizer = torch.optim.Adam(net.parameters(), config.lr)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

# prepare the Alphabets
alphabets = Alphabets(config.alphabets)
# prepare the tensor process
tensor_process = TensorProcess(alphabets)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(net, epoch):
    net = model_to_device(net)
    for e in range(epoch):
        net.train()
        epoch_loss = 0.0
        correct_count = 0
        total_count = 0
        for index, (data, label, label_length) in enumerate(train_loader):
            data = Variable(tensor_to_device(data))
            label = Variable(tensor_to_device(label)).int()
            # data = Variable(data).to(args.devices)
            # label = Variable(label).int().to(args.devices)
            b, l_lables = label.shape
            optimizer.zero_grad()
            outputs = net(data)
            l_outputs, b, p = outputs.size()
            loss = criterion(outputs, label,torch.IntTensor([l_outputs]*b), torch.IntTensor([l_lables]*b))
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            # evaluate the accuracy
            gt = label.view(-1).detach().cpu().numpy()
            gt = alphabets.encode(gt)
            gt_list = list()
            step = 4
            for index in range(len(label_length)):
                gt_list.append(gt[index*step:(index+1)*step])
            predict_labels = tensor_process.post_process_batch(outputs)
            correct_count += compare_str_list(gt_list, predict_labels)
            total_count += len(gt_list)
        eval_acc = eval_batch(net,e,test_loader_batch)
        if index % 1 == 0:
            acc = float(correct_count)/total_count
            print("epoch: {}/{}, loss:{}, train_acc:{}, eval_acc:{}, learning_rate: {}".format(e,epoch,epoch_loss,acc,eval_acc,scheduler.get_lr()))
        scheduler.step()
        if e % args.save_per_epoch == 0:
            save_model(net,e)

def eval(net, epoch):
    net.eval()
    net = model_to_device(net)
    correct = 0
    for index, (data, label, label_length) in enumerate(test_loader):
        data = Variable(tensor_to_device(data))
        label = Variable(tensor_to_device(label)).int()
        outputs = net(data)
        predict = tensor_process.post_process(outputs)
        gt = label.view(-1).detach().cpu().numpy()
        gt = alphabets.encode(gt)
        if predict == gt:
            correct += 1
    print("epoch: {}, eval acc: {}".format(epoch, float(correct)/len(test_loader)))

def eval_batch(net,epoch,loader):
    net.eval()
    net = model_to_device(net)
    correct = 0
    total_count=0
    for index, (data, label, label_length) in enumerate(loader):
        data = Variable(tensor_to_device(data))
        label = Variable(tensor_to_device(label)).int()
        outputs = net(data)
        predict_labels = tensor_process.post_process_batch(outputs)
        gt = label.view(-1).detach().cpu().numpy()
        gt = alphabets.encode(gt)
        gt_list = list()
        step = 4
        for index in range(len(label_length)):
            gt_list.append(gt[index * step:(index + 1) * step])
        correct += compare_str_list(gt_list, predict_labels)
        total_count += len(gt_list)
    return float(correct)/total_count

def main():
    if config.resume_model:
        resume_model(net, config.resume_model)
    else:
        net.apply(init_weights)
    train_epoch(net,config.epoch)

if __name__ == "__main__":
    main()