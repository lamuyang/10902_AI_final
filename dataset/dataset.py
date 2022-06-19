# coding:utf8
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
 

Labels = {"banana":0,"carrot":1,"dragonfruit":2,"guava":3,"peanut":4,"pumpkin":5,"soybean":6,"tomato":7,"bareland":8,"corn":9,"garlic":10,"pineapple":11,"rice":12,"sugarcane":13,}
 
 
class SeedlingData (data.Dataset):
 
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        self.transforms = transforms
 
        if self.test:
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
            self.imgs = imgs
        else:
            imgs_labels = [os.path.join(root, img) for img in os.listdir(root)]
            imgs = []
            for imglable in imgs_labels:
                for imgname in os.listdir(imglable):
                    imgpath = os.path.join(imglable, imgname)
                    imgs.append(imgpath)
            trainval_files, val_files = train_test_split(imgs, test_size=0.3, random_state=72)
            if train:
                self.imgs = trainval_files
            else:
                self.imgs = val_files
 
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_path=img_path.replace("\\",'/')
        if self.test:
            label = -1
        else:
            labelname = img_path.split('/')[-2]
            label = Labels[labelname]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label
 
    def __len__(self):
        return len(self.imgs)

