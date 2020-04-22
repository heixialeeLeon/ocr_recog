#coding=utf-8
import string
pool = ''
pool += string.ascii_letters
pool += "0123456789"
pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}"
pool += ' '
pool = '~'+pool
alphabets = pool

print(alphabets)
# data related parameters
# image_folder = '/data/Syntheic_Chinese/images'
# train_file_list = '/data/Syntheic_Chinese/train.txt'
# test_file_list = '/data/Syntheic_Chinese/test.txt'
image_folder = '/home/peizhao/data/ocr/English/gen_eng/data'
train_file_list = '/home/peizhao/data/ocr/English/gen_eng/label/train.txt'
test_file_list = '/home/peizhao/data/ocr/English/gen_eng/label/test.txt'
random_seed = 1111

# image_input_size = (160,32)
image_input_size = (1920,32)

# train related parameters
#resume_model ='checkpoint_ch/cn_v0.pth'
#resume_model ='checkpoint/recog_epoch_10.pth'
resume_model= ''

batch_size= 32
epoch = 50
save_per_epoch = 2
lr = 0.0001
step = [10, 20, 30]
display_interval = 20

# print(alphabets[1])
# print(alphabets[0])
# print(alphabets[2])
# print(len(alphabets))