import cv2
import numpy as np
from PIL import Image
import os
import copy
import csv

import torchvision
import torch
from .randaugment import RandAugment
from torch.utils import data

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
        self.strong_transfrom = copy.deepcopy(transform)
        self.strong_transfrom.transforms.insert(0, RandAugment(3,5))

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        out3 = self.strong_transfrom(inp)
        return out1, out2, out3


def get_raf(FER_img_folder_path,FER_TrainlabelPath,FER_TestlabelPath, n_labeled, transform_train=None, transform_val=None):
    
    train_labeled_idxs, train_unlabeled_idxs = data_split(FER_TrainlabelPath, int(n_labeled))

    train_labeled_dataset = Dataset_RAF_labeled(FER_img_folder_path, FER_TrainlabelPath, train_labeled_idxs, index=11, transform=transform_train)
    train_unlabeled_dataset = Dataset_RAF_unlabeled(FER_img_folder_path, FER_TrainlabelPath, train_unlabeled_idxs, index=11,transform=TransformTwice(transform_train))

    test_dataset = load_RAFDB(FER_img_folder_path ,FER_TestlabelPath, index=9, transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset)} #Test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    
def target_read(path):
    label_list = []
    with open(path) as f:
        img_label_list = f.read().splitlines()
    for info in img_label_list:
        _, label_name = info.split(' ')
        label_list.append(int(label_name))
    return label_list

def data_split(filename, n_labeled):
    SEED = 5
    labels = target_read(filename)
    labels = np.array(labels)
    #print(len(labels))
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(1,8):
        #num = 0
        idxs = np.where(labels == i)[0]
        #print(' label %d nums:' %i, idxs.shape)
        np.random.seed(SEED)
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:int(n_labeled/7)])
        train_unlabeled_idxs.extend(idxs[int(n_labeled/7):])

    np.random.seed(SEED)
    np.random.shuffle(train_labeled_idxs)
    np.random.seed(SEED)
    np.random.shuffle(train_unlabeled_idxs)

    print('train_labeled_idxs: ', len(train_labeled_idxs))
    print('train_unlabeled_idxs: ', len(train_unlabeled_idxs))
    return train_labeled_idxs, train_unlabeled_idxs

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)

class load_RAFDB(data.Dataset):
    def __init__(self,dataRoot,labelPath,index,transform=None):
        with open(labelPath,'r') as f:
            labels=f.readlines()
        self.tranform=transform
        self.dataRoot=dataRoot
        self.imgList=[]
        self.labelList=[]
        self.index=index
        for xx in labels:
            x =xx.split()
            self.imgList.append(x[0])
            self.labelList.append(int(x[1])-1)
        self.label = np.array(self.labelList)

    def __getitem__(self,index):
        imgName=self.imgList[index]
        imgName_=list(imgName)
        imgName_.insert(self.index,'_aligned')
        imgName="".join(imgName_)
        imgPath=os.path.join(self.dataRoot,imgName)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label=torch.IntTensor([self.labelList[index]]).squeeze(-1)
        return img,label

    def __len__(self):
        return len(self.imgList)
    
class Dataset_RAF_labeled(load_RAFDB):
    def __init__(self,dataRoot, labelPath, indexs,index, transform=None):
        super(Dataset_RAF_labeled, self).__init__(dataRoot,labelPath, index,transform=transform)

        if indexs is not None:
            self.imgList = np.array(self.imgList)[indexs]
            self.labelList = np.array(self.labelList)[indexs]

    def __getitem__(self, index):
        imgName=self.imgList[index]
        imgName_=list(imgName)
        imgName_.insert(self.index,'_aligned')
        imgName="".join(imgName_)
        imgPath=os.path.join(self.dataRoot,imgName)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label=torch.IntTensor([self.labelList[index]]).squeeze(-1)
        return img,label

class Dataset_RAF_unlabeled(Dataset_RAF_labeled):
    def __init__(self, dataRoot,labelPath, indexs, index,transform=None):
        super(Dataset_RAF_unlabeled, self).__init__(dataRoot,labelPath,indexs, index,transform=transform)
        self.labelList = np.array([-1 for i in range(len(self.labelList))])
        
if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torch.utils.data as data
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # RAFDB
    n_labeled = 500
    FER_TestlabelPath = "/home/dataset/FaceData/RAF-DB/label/test.txt"
    FER_TrainlabelPath = "/home/dataset/FaceData/RAF-DB/label/train.txt"
    FER_img_folder_path ="/home/dataset/FaceData/RAF-DB/aligned/" 
    train_labeled_set, train_unlabeled_set, test_set = get_raf(FER_img_folder_path,FER_TrainlabelPath,FER_TestlabelPath, n_labeled, transform_train=transform_train, transform_val=transform_val)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=64, shuffle=True, num_workers=10, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=64, shuffle=True, num_workers=10, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=10)

    for img,target in labeled_trainloader:
        print(img.shape)
        print(target.shape)
        
    for img,target in unlabeled_trainloader:
        print(img[0].shape)
        print(target.shape)
        
    for img,target in test_loader:
        print(img.shape)
        print(target.shape)