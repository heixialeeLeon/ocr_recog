alphabets = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
image_folder = '/data/vott-csv-export'
train_test_rate = 0.95
image_input_size = (150,53)

resume_model ='checkpoint/recog_v2.pth'
#resume_model ='checkpoint/recog_epoch_100.pth'

#trainning parameters
epoch = 600
lr = 0.0001
step = [200, 400]