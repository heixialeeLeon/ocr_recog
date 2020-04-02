alphabets = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#/_'
#image_folder = '/data/vott-csv-export'
#image_folder = '/data/captcha/航华验证码'
train_folder = '/home/peizhao/data/work/cell/cell_train'
test_folder = '/home/peizhao/data/work/cell/cell_test'
# train_folder = '/data/captcha/train_total'
# test_folder = '/data/captcha/huahang/test'
train_test_rate = 0.9
image_input_size = (200,20)
#image_input_size = (150,53)

random_seed = 1111

#resume_model ='checkpoint/recog_v2.pth'
resume_model ='checkpoint/cell_epoch_200.pth'
#resume_model =''


#trainning parameters
epoch = 300
lr = 0.00001
step = [50,100,200]