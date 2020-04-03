import cv2
from utils.img_show import *
import os

folder = "/home/peizhao/data/work/cell/temp"
gt = ['1.png','2.png','3.png','4.png']
gen = ['1_1.png','2_1.png','3_1.png','4_1.png']

for index in range(len(gt)):
    src_file = os.path.join(folder,gt[index])
    dst_file = os.path.join(folder,gen[index])
    img1 = cv2.imread(src_file)
    img2 = cv2.imread(dst_file)
    img1 = cv2.resize(img1,(320,32))
    img2 = cv2.resize(img2,(320,32))
    im = np.vstack((img1,img2))
    # cv2.imwrite("result_{}.png".format(index),im)
    cv2.imshow("test",im)
    cv2.waitKey(0)

# target_height =32
# for index in range(len(gt)):
#     src_file = os.path.join(folder,gt[index])
#     dst_file = os.path.join(folder,gen[index])
#     img1 = cv2.imread(src_file)
#     h,w,_= img1.shape
#     scale = target_height/h
#     new_size = (int(w*scale),int(h*scale))
#     img1 = cv2.resize(img1,new_size)
#
#     img2 = cv2.imread(dst_file)
#     h,w,_= img2.shape
#     scale = target_height/h
#     new_size = (int(w*scale),int(h*scale))
#     img2 = cv2.resize(img2, new_size)
#     # img1 = cv2.resize(img1,(200,20))
#     # img2 = cv2.resize(img2,(200,20))
#     # cv2.imwrite("result_{}.png".format(index),im)
#     cv2.imshow("img1",img1)
#     cv2.imshow("img2",img2)
#     cv2.waitKey(0)
