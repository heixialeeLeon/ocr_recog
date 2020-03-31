import os
from utils.list_utils import split_with_shuffle
import shutil

def split_train_test(src_folder, train_folder, test_folder):
    image_files = os.listdir(src_folder)
    train, test = split_with_shuffle(image_files,0.9)
    for item in train:
        src_file = os.path.join(src_folder,item)
        dst_file = os.path.join(train_folder,item)
        shutil.copyfile(src_file,dst_file)

    for item in test:
        src_file = os.path.join(src_folder,item)
        dst_file = os.path.join(test_folder,item)
        shutil.copyfile(src_file,dst_file)

    print("finished, total copy: {} files".format(len(train)+len(test)))


if __name__ == "__main__":
    train_folder = "/data/captcha/huahang/train"
    test_folder = "/data/captcha/huahang/test"
    src_folder = "/data/captcha/航华验证码"
    split_train_test(src_folder,train_folder,test_folder)