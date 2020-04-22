from models.vgg16_crnn import VGG16_CRNN
import torch

model = VGG16_CRNN(100)
save_model_name = "aa.pth"

torch.save(model, save_model_name)

net =torch.load(save_model_name)
print(net)