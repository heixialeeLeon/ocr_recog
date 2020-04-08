import cv2
from pathlib import Path
import os
import random
import numpy as np

def split(img_path):
    images = os.listdir(img_path)
    for item in images:
        file_name = item.split('.')[0]
        src = os.path.join(img_path,item)
        dst1 = os.path.join(img_path,file_name+'_1'+'.png')
        dst2 = os.path.join(img_path,file_name+'_2'+'.png')
        im =cv2.imread(src)
        im = cv2.resize(im,(320,64))
        im1 = im[:32,]
        im2 = im[32:,]
        # cv2.imshow("a",im)
        # cv2.imshow("1",im1)
        # cv2.imshow("2",im2)
        cv2.imwrite(dst1,im1)
        cv2.imwrite(dst2,im2)

def random_scale(img_path):
    images = os.listdir(img_path)
    for item in images:
        src = os.path.join(img_path, item)
        im = cv2.imread(src,0)
        im = cv2.resize(im, (320, 32))
        h,w = im.shape
        scale_w = random.randint(0,40)
        new_w = w - scale_w
        print(new_w)
        im_new = im[:,:new_w]
        im_new = cv2.resize(im_new, (320, 32))
        final = np.vstack((im,im_new))
        cv2.imshow("aa",final)
        cv2.waitKey(0)

def salt_and_pepper_noise(img, proportion=0.02):
    noise_img =img
    height,width =noise_img.shape[0],noise_img.shape[1]
    num = int(height*width*proportion)#多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0,width-1)
        h = random.randint(0,height-1)
        if random.randint(0,1) ==0:
            noise_img[h,w] =0
        else:
            noise_img[h,w] = 255
    return noise_img

def process(img_path):
    images = os.listdir(img_path)
    for item in images:
        src = os.path.join(img_path, item)
        im = cv2.imread(src,0)
        im_new = salt_and_pepper_noise(im)
        cv2.imshow("aa",im_new)
        cv2.waitKey(0)


if __name__ =="__main__":
    #split("/home/peizhao/data/work/cell/temp1")
    # random_scale("/home/peizhao/data/work/cell/temp1")
    process("/home/peizhao/data/work/cell/temp1")