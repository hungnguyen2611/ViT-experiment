from torch.utils.data import Dataset
import glob
from PIL import Image
import os
import numpy as np

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, additional_dir=None, transform=None):
        self.img_dir = glob.glob(img_dir + '/*/*')
        if additional_dir:
            self.img_dir = self.img_dir + glob.glob(additional_dir+'/*/*')
        self.class_dict = {
            "p_imgs": 0,
            "org_imgs": 1,
            "pet_imgs": 2
        }
        self.transform = transform
        
    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir[idx])
        class_name = os.path.basename(os.path.dirname(self.img_dir[idx]))
        label = self.class_dict[class_name]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = np.array(label, dtype=np.int_)
        return img, label