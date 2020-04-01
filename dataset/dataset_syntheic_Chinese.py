from pathlib import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch.nn.utils.rnn as rnn
from utils.alphabets import Alphabets
from utils.img_show import *
from utils.file_utils import get_file_list
from utils.list_utils import split_with_shuffle
import config.config_syntheic_Chinese as config
import os

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
        image = cv2.resize(image,self.size)
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

train_dataset = DatasetFromTextFile(config.train_file_list)
test_dataset = DatasetFromTextFile(config.test_file_list)

if __name__ == "__main__":
    train_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=8, shuffle=False,
                              collate_fn=default_collate_fn)
    for data, label, label_length in train_loader:
        images = [item for item in data[:, ]]
        #CV2_showTensors_unsqueeze(images)
        #CV2_showTensors(img)
        print(data.shape)
        #print(label.shape)
        #print(label_length)