from pathlib import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch.nn.utils.rnn as rnn
from utils.alphabets import Alphabets
from utils.img_show import *
from utils.file_utils import get_file_list
from utils.list_utils import split_with_shuffle
from datasets.tools import PAD
import config.config_syntheic_Chinese as config
from utils.alphabets import Alphabets_Chinese
import os
from lib.utils import view

input_transform = Compose([
	ToTensor(),
	# Normalize([.485, .456, .406], [.229, .224, .225]),
])

class DatasetFromTextFile(Dataset):
    def __init__(self,text_file, image_folder=config.image_folder, size=config.image_input_size):
        with open(text_file,'r') as f:
            self.file_list = f.readlines()
        self.length = len(self.file_list)
        self.alphabets = Alphabets(config.alphabets)
        self.size = size
        self.image_folder = image_folder

    def __getitem__(self, index):
        file_name = str(self.file_list[index])
        file_name,label = file_name.split(' ')
        label = label.strip()
        file_name = os.path.join(self.image_folder,file_name)
        image = cv2.imread(file_name,0)
        # image = cv2.resize(image,self.size)
        image = input_transform(image)
        target = torch.tensor(self.alphabets.decode(label))
        return image,target

    def __len__(self):
        return self.length

def default_collate_fn(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    label_length = [len(item) for item in label]
    padding_label = rnn.pad_sequence(label, batch_first=True, padding_value=0)
    data = torch.stack(data, dim=0)
    return data, padding_label, label_length

class AlginCollate(object):
    def __init__(self, img_h, img_w, channel=3):
        self.pad = PAD((channel,img_h,img_w))

    def __call__(self,batch):
        data = [self.pad(item[0]) for item in batch]
        # data = [item[0] for item in batch]
        label = [item[1] for item in batch]
        label_length = [len(item) for item in label]
        padding_label = rnn.pad_sequence(label, batch_first=True, padding_value=0)
        data = torch.stack(data, dim=0)
        return data, padding_label, label_length

train_dataset = DatasetFromTextFile(config.train_file_list)
test_dataset = DatasetFromTextFile(config.test_file_list)
align_collate = AlginCollate(config.image_input_size[1],config.image_input_size[0],1)

if __name__ == "__main__":
    alphabets = Alphabets_Chinese(config.alphabets)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=8, shuffle=True,
                              collate_fn=align_collate)
    for data, label, label_length in train_loader:
        images = [item for item in data[:, ]]
        gt_list = list()
        label = label.detach().cpu().numpy()
        for item in label[:, ]:
            gt = alphabets.encode(item)
            gt =''.join(gt).replace('~', '')
            gt_list.append(gt)
        print(gt_list)
        #CV2_showTensors_unsqueeze_gray(images,direction=1,timeout=0)
        view.show_tensor(images, direction=view.DIRECTION_VERTICAL, timeout=0)
        # print(data.shape)
        # print(label.shape)
        #print(label_length)