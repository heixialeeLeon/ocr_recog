import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tvF
from skimage import util
import sys

def CV2_showPILImage(pil_image, timeout =1000):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow("test",cv_image)
    cv2.waitKey(timeout)

def CV2_showPILImage_Float(pil_image, timeout=1000):
    im = np.array(pil_image)
    im = util.img_as_ubyte(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow("test",im)
    cv2.waitKey(timeout)

def CV2_showPILImage_List(imgs, timeout=1000):
    first_image = True
    img_show = None
    for item in imgs:
        item = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2BGR)
        if first_image:
            img_show = item
            first_image = False
        else:
            img_show = np.hstack((img_show, item))
    cv2.imshow("test", img_show)
    cv2.waitKey(timeout)

def CV2_showPILImages(*imgs,timeout=1000):
    first_image = True
    img_show = None
    for item in imgs:
        item = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2BGR)
        if first_image:
            img_show = item
            first_image = False
        else:
            img_show=np.hstack((img_show,item))
    cv2.imshow("test", img_show)
    cv2.waitKey(timeout)

def CV2_showTensors(*tensors, timeout=1000,direction=0):
    first_image = True
    img_show = None
    for item in tensors:
        item = torch.squeeze(item).data.cpu().numpy().transpose(1,2,0)
        item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        if first_image:
            img_show = item
            first_image = False
        else:
            if direction == 0:
                img_show = np.hstack((img_show, item))
            else:
                img_show = np.vstack((img_show, item))
    cv2.imshow("CV2_show", img_show)
    cv2.waitKey(timeout)

def CV2_showTensors_unsqueeze(tensors, timeout=1000):
    first_image = True
    img_show = None
    for item in tensors:
        item = item.data.cpu().numpy().transpose(1,2,0)
        item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        if first_image:
            img_show = item
            first_image = False
        else:
            img_show = np.hstack((img_show, item))
    cv2.imshow("CV2_show", img_show)
    cv2.waitKey(timeout)

def CV2_showTensors_unsqueeze_gray(tensors, timeout=1000,direction=0):
    first_image = True
    img_show = None
    for item in tensors:
        item = item.data.cpu().numpy().transpose(1,2,0)
        if first_image:
            img_show = item
            first_image = False
        else:
            if direction == 0:
                img_show = np.hstack((img_show, item))
            else:
                img_show = np.vstack((img_show, item))
    cv2.imshow("CV2_show", img_show)
    cv2.waitKey(timeout)

def CV2_showTensors_Resize(*tensors, resize,timeout=1000):
    first_image = True
    img_show = None
    for item in tensors:
        item = torch.squeeze(item).data.cpu().numpy().transpose(1,2,0)
        item = cv2.resize(item, dsize=resize,interpolation=cv2.INTER_LINEAR)
        item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
        if first_image:
            img_show = item
            first_image = False
        else:
            img_show = np.hstack((img_show, item))
    cv2.imshow("CV2_show", img_show)
    cv2.waitKey(timeout)

def PIL_ShowPILImage2(pil_image1, pil_image2):
    fig = plt.figure(figsize=(1,2))
    fig.set_size_inches(10,5)
    fig.add_subplot(1,2,1)
    plt.imshow(pil_image1)
    fig.add_subplot(1,2,2)
    plt.imshow(pil_image2)
    plt.show()

def PIL_ShowPILImage3(pil_image1,pil_image2,pil_image3):
    fig = plt.figure(figsize=(1, 3))
    fig.set_size_inches(10, 5)
    fig.add_subplot(1, 3, 1)
    plt.imshow(pil_image1)
    fig.add_subplot(1, 3, 2)
    plt.imshow(pil_image2)
    fig.add_subplot(1, 3, 3)
    plt.imshow(pil_image3)
    plt.show()

def PIL_ShowTensor(tensor):
    pil_img = tvF.to_pil_image(tensor)
    fig = plt.figure()
    plt.imshow(pil_img)
    plt.show()

def PIL_ShowTensor_Timeout(tensor, timeout=1000):

    class CloseEvent(object):

        def __init__(self):
            self.first = True

        def __call__(self):
            if self.first:
                self.first = False
                return
            plt.close()

    pil_img = tvF.to_pil_image(tensor)
    fig = plt.figure()
    timer=fig.canvas.new_timer(interval=timeout)
    timer.add_callback(CloseEvent())
    plt.imshow(pil_img)
    timer.start()
    plt.show()

def PIL_ShowTensor2(tensor1, tensor2):
    pil_img1 = tvF.to_pil_image(tensor1)
    pil_img2 = tvF.to_pil_image(tensor2)
    PIL_ShowPILImage2(pil_img1,pil_img2)

def PIL_ShowTensor3(tensor1, tensor2,tensor3):
    pil_img1 = tvF.to_pil_image(tensor1)
    pil_img2 = tvF.to_pil_image(tensor2)
    pil_img3 = tvF.to_pil_image(tensor3)
    PIL_ShowPILImage3(pil_img1,pil_img2,pil_img3)