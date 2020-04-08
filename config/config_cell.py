alphabets = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#/_'
#image_folder = '/data/vott-csv-export'
#image_folder = '/data/captcha/航华验证码'
train_folder = '/home/peizhao/data/work/cell/0403/cell_train'
test_folder = '/home/peizhao/data/work/cell/0403/cell_test'
#test_folder = '/home/peizhao/data/work/cell/temp1'
# train_folder = '/data/captcha/train_total'
# test_folder = '/data/captcha/huahang/test'
train_test_rate = 0.9
image_input_size = (320,32)
#image_input_size = (150,53)

random_seed = 1111

#resume_model ='checkpoint/recog_v2.pth'
#resume_model ='checkpoint_cell/cell_simple_crnn_v0.pth'
#resume_model ='checkpoint_cell/cell_simple_crnn_v2.pth'
#resume_model ='checkpoint/cell_epoch_45.pth'
resume_model ='checkpoint_cell/cell_resnet_v0.pth'
#resume_model = ''

#trainning parameters
epoch = 100
lr = 0.0001
step = [20,40,80]
save_per_epoch=5
batch_size =32