#import cv2
import numpy as np
from PIL import Image
import os,sys
import copy
import csv
from torchvision import datasets
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

FERPlus_img_folder_path ="/home/dataset/FaceData/FERPlus/data-aligned/"

def get_affectnet8( n_labeled, transform_train=None, transform_val=None):
    
    
    img_folder = "/home/dataset/FaceData/AffectNet/split"
    TrainlabelPath = "/home/dataset/FaceData/AffectNet/training.npy"
    TestlabelPath = "/home/dataset/FaceData/AffectNet/validation.npy"
    FER_img_folder_path ="/home/dataset/FaceData/AffectNet/align/"   
    base_dataset = datasets.ImageFolder(img_folder, transform=None)
    
    train_labeled_idxs, train_unlabeled_idxs = data_split8(base_dataset.targets, int(n_labeled))

    train_labeled_dataset = AffectNet_labeled( img_folder,train_labeled_idxs,  transform=transform_train)
    train_unlabeled_dataset = AffectNet_unlabeled( img_folder,train_unlabeled_idxs, transform=TransformTwice(transform_train))

    test_dataset =  load_AffectNet(TestlabelPath,FER_img_folder_path,transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset)} #Test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_affectnet7( n_labeled, transform_train=None, transform_val=None):
    
    
    img_folder = "/home/dataset/FaceData/AffectNet/split_7"
    TrainlabelPath = "/home/dataset/FaceData/AffectNet/training.npy"
    TestlabelPath = "/home/dataset/FaceData/AffectNet/validation.npy"
    FER_img_folder_path ="/home/dataset/FaceData/AffectNet/align/"  
    base_dataset = datasets.ImageFolder(img_folder, transform=None)
    
    train_labeled_idxs, train_unlabeled_idxs = data_split7(base_dataset.targets, int(n_labeled))

    train_labeled_dataset = AffectNet_labeled( img_folder,train_labeled_idxs,  transform=transform_train)
    train_unlabeled_dataset = AffectNet_unlabeled( img_folder,train_unlabeled_idxs, transform=TransformTwice(transform_train))

    test_dataset =  load_AffectNet_7(TestlabelPath,FER_img_folder_path,transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset)} #Test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def data_split8(labels, n_labeled):
    labels = np.array(labels)
    #print(len(labels))
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(8):
        #num = 0
        idxs = np.where(labels == i)[0]
        #print(' label %d nums:' %i, idxs.shape)
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:int(n_labeled/8)])
        train_unlabeled_idxs.extend(idxs[int(n_labeled/8):])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    print('train_labeled_idxs: ', len(train_labeled_idxs))
    print('train_unlabeled_idxs: ', len(train_unlabeled_idxs))
    return train_labeled_idxs, train_unlabeled_idxs

def data_split7(labels, n_labeled):
    labels = np.array(labels)
    #print(len(labels))
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(7):
        #num = 0
        idxs = np.where(labels == i)[0]
        #print(' label %d nums:' %i, idxs.shape)
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:int(n_labeled/7)])
        train_unlabeled_idxs.extend(idxs[int(n_labeled/7):])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    print('train_labeled_idxs: ', len(train_labeled_idxs))
    print('train_unlabeled_idxs: ', len(train_unlabeled_idxs))
    return train_labeled_idxs, train_unlabeled_idxs

    
class AffectNet_labeled(datasets.ImageFolder):
    def __init__(self,img_folder,indexs, transform=None):
        super(AffectNet_labeled, self).__init__(img_folder,transform=transform)

        if indexs is not None:
            self.samples = np.array(self.samples)[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target=np.array(target).astype(int)
        return sample, target
    
class AffectNet_unlabeled(AffectNet_labeled):
    def __init__(self,  img_folder,indexs, transform=None):
        super(AffectNet_unlabeled, self).__init__(img_folder,indexs, transform=transform)
        self.targets = np.array([-1 for i in range(len(self.targets))])
        self.samples[1] = np.array([-1 for i in range(len(self.samples[1]))])
        
class load_AffectNet(data.Dataset):
    def __init__(self,label_npy_path,dataRoot,transform=None):
        self.labelList=np.load(label_npy_path)
        self.tranform=transform
        self.dataRoot=dataRoot
        #self.labelList=np.array(self.labelList)
        
    def __getitem__(self,index):
        all=self.labelList[index]
        imgName=all[0]
        imgPath=os.path.join(self.dataRoot,imgName)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label_raw=int(all[1])
        label=torch.IntTensor([label_raw])
        return img,label

    def __len__(self):
        return len(self.labelList)

class load_AffectNet_7(data.Dataset):
    def __init__(self,label_npy_path,dataRoot,transform=None):
        self.labelList=np.load(label_npy_path)
        self.tranform=transform
        self.labelList=self.remove_comtempt()
        #self.labelList=np.array(self.remove_comtempt())
        self.dataRoot=dataRoot
    def remove_comtempt(self):
        temp_list_7=[]
        for item in self.labelList:
            if int(item[1])!=7:
                temp_list_7.append(item)
        # self.labelList=temp_list_7
        return np.array(temp_list_7)
    def __getitem__(self,index):
        all=self.labelList[index]
        imgName=all[0]
        imgPath=os.path.join(self.dataRoot,imgName)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label_raw=int(all[1])
        label=torch.IntTensor([label_raw])
        return img,label

    def __len__(self):
        return len(self.labelList)
        
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
    n_labeled = 96

    train_labeled_set, train_unlabeled_set, test_set = get_affectnet8( n_labeled, transform_train=transform_train, transform_val=transform_val)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=64, shuffle=True, num_workers=10, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=64, shuffle=True, num_workers=10, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=10)

    for img,target in labeled_trainloader:
        print(img.shape)
        #print(target)
        print(target.shape)
        
    for img,target in unlabeled_trainloader:
        print(img[0].shape)
        print(target.shape)
        
    for img,target in test_loader:
        print(img.shape)
        print(target.shape)