import os
import os.path as osp
import PIL.Image as Image
from torch.utils.data import Dataset
import numpy as np


CLASSES = ['Open-top container', 'Closed double container', 'Closed single container', 'Other types of carriage', 'Sandy soil carriage', 
               'Steel coil carriage', 'Tank container', 'Carriage with big tarp', 'Carriage with small tarp']
CLASSES_cn = ['敞顶集装箱','封闭双集装箱', '封闭单集装箱', '其他类型车厢','沙土车厢', '钢卷车厢', '罐式集装箱','大篷布车厢','小篷布车厢']
RESULTS = ['is normal', 'occurs litters', 'occurs persons', 'occurs container broken', 'occurs pits', 
               'misses tanker covers', 'occurs tarp ropes broken', 'occurs tarp broken', 'occurs side ropes decoupling', 'occurs side ropes broken',
               'occurs standing water']
RESULTS_cn = ['无异常', '有杂物', '有闲杂人员', '箱体破损', '车厢凹陷', '顶盖丢失', '绳网破损', '篷布破损', '边绳脱钩', '边绳断裂', '车厢积水']


class Datasets(object):
    def __init__(self, file_path, transform=None): 
        self.file_path = file_path
        self.transform = transform
        self.imgs, self.labels = [], []
        if osp.exists(self.file_path):
            with open(self.file_path) as f:
                lines = f.readlines() # jpg #@# label
                for line in lines:
                    jpg, label = line.split('#@#')
                    self.imgs.append(jpg)
                    for i, cls_name in enumerate(RESULTS_cn):
                        if cls_name in label:
                            self.labels.append(i)
        else:
            raise Exception(f"In valid dataset path {self.file_path}")

        self.length = len(self.labels)
    
    def __len__(self):
        return self.length

    
    def __getitem__(self, index):
        label       = self.labels[index]
        img_path    = self.imgs[index].replace('/home/huojian/yolov7', '..')
        img         = Image.open(img_path).convert('RGB')
        img         = self.transform(img)
        return img, label


class MergedDataset(Dataset):

    """
    Takes two datasets (dataset1, dataset2) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, dataset1, dataset2):

        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):

        if index < len(self.dataset1):
            img, label = self.dataset1[index]
        else:
            img, label = self.dataset2[index - len(self.dataset1)]

        return img, label

    def __len__(self):
        return len(self.dataset2) + len(self.dataset1)