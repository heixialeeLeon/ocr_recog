alphabets = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#image_folder = '/data/vott-csv-export'
image_folder = '/data/captcha/航华验证码'
train_folder = '/data/captcha/huahang/train'
test_folder = '/data/captcha/huahang/test'
train_test_rate = 0.9
image_input_size = (150,53)

#resume_model ='checkpoint/recog_v2.pth'
resume_model ='checkpoint/recog_epoch_700.pth'
#resume_model =''

#trainning parameters
epoch = 800
lr = 0.0001
step = [400, 600]