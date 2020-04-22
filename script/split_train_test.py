import os
from utils.list_utils import split_with_shuffle
import shutil
from tqdm import *

def split_train_test(src_folder, train_folder, test_folder,rate=0.9):
    image_files = os.listdir(src_folder)
    train, test = split_with_shuffle(image_files,rate)
    print("start to copy train datasets")
    for index in tqdm(range(len(train))):
        src_file = os.path.join(src_folder,train[index])
        dst_file = os.path.join(train_folder,train[index])
        shutil.copyfile(src_file,dst_file)

    print("start to copy test datasets")
    for index in tqdm(range(len(test))):
        src_file = os.path.join(src_folder,test[index])
        dst_file = os.path.join(test_folder,test[index])
        shutil.copyfile(src_file,dst_file)

    print("finished, total copy: {} files".format(len(train)+len(test)))

def split_train_test_label(label_file, output_folder, rate=0.9):
    with open(label_file,'r') as f:
        label_list = f.readlines()
    train_file = os.path.join(output_folder,"train.txt")
    test_file = os.path.join(output_folder,"test.txt")
    train, test = split_with_shuffle(label_list,rate)
    print("start to generate the train label")
    with open(train_file,'w') as f:
        for index in tqdm(range(len(train))):
            f.write(train[index])
    print("start to generate the test label")
    with open(test_file, 'w') as f:
        for index in tqdm(range(len(test))):
            f.write(test[index])

# if __name__ == "__main__":
#     # train_folder = "/data/captcha/huahang/train"
#     # test_folder = "/data/captcha/huahang/test"
#     # src_folder = "/data/captcha/航华验证码"
#     train_folder = "/home/peizhao/data/work/cell/0403/cell_train"
#     test_folder = "/home/peizhao/data/work/cell/0403/cell_test"
#     src_folder = "/home/peizhao/data/work/cell/0403/raw_data"
#     split_train_test(src_folder,train_folder,test_folder,0.95)

if __name__ == "__main__":
    # label_file = "/home/peizhao/data/ocr/generate_1/label/label.txt"
    # output_folder = "/home/peizhao/data/ocr/generate_1/label"
    # label_file = "/home/peizhao/data/ocr/English/gen_eng/label/label.txt"
    # output_folder = "/home/peizhao/data/ocr/English/gen_eng/label"
    label_file = "/home/peizhao/data/ocr/handwrite/company/label/label.txt"
    output_folder = "/home/peizhao/data/ocr/handwrite/company/label"
    split_train_test_label(label_file,output_folder,rate=0.95)

