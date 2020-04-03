import os
from utils.list_utils import split_with_shuffle
import shutil
from tqdm import *

def split_train_test(src_folder, train_folder, test_folder,rate=0.9):
    image_files = os.listdir(src_folder)
    train, test = split_with_shuffle(image_files,rate)
    print("start to copy train dataset")
    for index in tqdm(range(len(train))):
        src_file = os.path.join(src_folder,train[index])
        dst_file = os.path.join(train_folder,train[index])
        shutil.copyfile(src_file,dst_file)
    # for item in train:
    #     src_file = os.path.join(src_folder,item)
    #     dst_file = os.path.join(train_folder,item)
    #     shutil.copyfile(src_file,dst_file)

    print("start to copy test dataset")
    for index in tqdm(range(len(test))):
        src_file = os.path.join(src_folder,test[index])
        dst_file = os.path.join(test_folder,test[index])
        shutil.copyfile(src_file,dst_file)
    # for item in test:
    #     src_file = os.path.join(src_folder,item)
    #     dst_file = os.path.join(test_folder,item)
    #     shutil.copyfile(src_file,dst_file)

    print("finished, total copy: {} files".format(len(train)+len(test)))

if __name__ == "__main__":
    # train_folder = "/data/captcha/huahang/train"
    # test_folder = "/data/captcha/huahang/test"
    # src_folder = "/data/captcha/航华验证码"
    train_folder = "/home/peizhao/data/work/cell/0403/cell_train"
    test_folder = "/home/peizhao/data/work/cell/0403/cell_test"
    src_folder = "/home/peizhao/data/work/cell/0403/raw_data"
    split_train_test(src_folder,train_folder,test_folder,0.95)

