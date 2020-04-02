alphabets = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#image_folder = '/data/vott-csv-export'
#image_folder = '/data/captcha/航华验证码'
train_folder = '/data/captcha/train_total'
test_folder = '/data/captcha/huahang/test'
# train_folder = '/home/peizhao/data/captcha/peizhao_generator/train'
# test_folder = '/home/peizhao/data/captcha/peizhao_generator/test'
train_test_rate = 0.9
image_input_size = (150,53)

resume_model =''
#resume_model ='checkpoint/recog_epoch_700.pth'
#resume_model ='checkpoint/captcha_epoch_100.pth'

#trainning parameters
epoch = 350
lr = 0.0001
step = [50, 150, 250]
display_interval = 20