import os
import torch.utils.data as data
from PIL import Image
import re

class CatDogDataset(data.Dataset):
    def __init__(self, files, path, mode='train', transform = None):
        self.files = files
        self.path = path
        self.mode = mode
        self.transform = transform
        self.label_mapping_idx = {'cat': 0, 'dog': 1}
        self.idx_mapping_label = {0: 'cat', 1: 'dog'}

        self.labels = []
        if (self.mode == 'train'):
            for idx in range(len(self.files)):
                match = re.match(r'([a-zA-Z]+)\.\d+\.jpg', self.files[idx])
                label = self.label_mapping_idx[match.group(1)]
                self.labels.append(label)
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.files[idx]))
        
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy().astype('float32')
            return img, self.labels[idx], self.files[idx]
        else:
            match = re.search(r'(\d+)', self.files[idx])
            idx = match.group(1)
            img = img.numpy().astype('float32')
            return img, int(idx)