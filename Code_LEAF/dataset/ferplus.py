#import cv2
import numpy as np
from PIL import Image
import os,sys
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

FERPlus_img_folder_path ="/home/dataset/FaceData/FERPlus/data-aligned/"

def get_ferplus( n_labeled, transform_train=None, transform_val=None):
    
    base_dataset = FERPlus(FERPlus_img_folder_path, 'train',transform=None)
        
    train_labeled_idxs, train_unlabeled_idxs = data_split(base_dataset.labels, int(n_labeled))

    train_labeled_dataset = FERPlus_labeled( train_labeled_idxs,  transform=transform_train)
    train_unlabeled_dataset = FERPlus_unlabeled( train_unlabeled_idxs, transform=TransformTwice(transform_train))

    test_dataset = FERPlus(FERPlus_img_folder_path,'test', transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset)} #Test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    
def data_split(labels, n_labeled):
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

class FERPlus(data.Dataset):
    """ FERPlus dataset"""
    def __init__(self, data_path, phase = 'train', mode = 'majority', transform = None, lmk = False):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        self.lmk = lmk
        self.mode = mode
        self.EMOTIONS = {0:"neutral", 1:"happiness", 2:"surprise", 3:"sadness", 4:"anger", 5:"disgust", 6:"fear", 7:"contempt"}
        self.EMOTIONS2Index = {"neutral":0, "happiness":1, "surprise":2, "sadness":3, "anger":4, "disgust":5, "fear":6, "contempt":7}
        #'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'

        # read annotations
        self.file_paths = []
        self.labels = []
        if phase == 'train':
            self.get_labels_('FER2013Train')
            self.get_labels_('FER2013Valid')
        else:
            self.get_labels_('FER2013Test')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path)
        label = self.labels[idx]
       
        if self.transform is not None:
            image = self.transform(image)
        
        if self.lmk:
            lmk_path = path.replace('Image/aligned','Landmarks')
            lmk_path = lmk_path.replace('.png', '_landmarks.txt')
            lmks = self.get_landmark_(lmk_path)
            return image, lmks, path, label, idx
        else:
            return image, label
        
    def get_landmark_(self, lmk_path):
        return 0

    def get_labels_(self, subfoler):
        with open(os.path.join(self.data_path, subfoler, 'label.csv')) as csvfile: 
            emotion_label = csv.reader(csvfile)
            for row in emotion_label: 
                emotion_raw = list(map(float, row[2:len(row)]))
                emotion = self.process_data_(emotion_raw, self.mode) 
                idx = np.argmax(emotion)
                if idx > 7: # not unknown or non-face 
                    continue
                if self.mode=='majority':
                    self.labels.append(idx)
                else:
                    self.labels.append(emotion)
                self.file_paths.append(os.path.join(self.data_path, subfoler,row[0]))

                
    def process_data_(self, emotion_raw, mode):
        '''
        Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:
        Majority: return the emotion that has the majority vote, or unknown if the count is too little.
        Probability or Crossentropty: convert the count into probability distribution.abs
        Multi-target: treat all emotion with 30% or more votes as equal.
        '''        
        size = len(emotion_raw)
        emotion_unknown     = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal) 
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size 

        if mode == 'majority': 
            # find the peak value of the emo_raw list 
            maxval = max(emotion_raw) 
            if maxval > 0.5*sum_list: 
                emotion[np.argmax(emotion_raw)] = maxval 
            else: 
                emotion = emotion_unknown   # force setting as unknown 
        elif (mode == 'probability') or (mode == 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw) 
                for i in range(size): 
                    if emotion_raw[i] == maxval: 
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share same number of max votes 
                            valid_emotion = False
                            if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown   # force setting as unknown 
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size): 
                if emotion_raw[i] >= threshold*sum_list: 
                    emotion[i] = emotion_raw[i] 
            if sum(emotion) <= 0.5 * sum_list: # less than 50% of the votes are integrated, we discard this example 
                emotion = emotion_unknown   # set as unknown 
                                
        return [float(i)/sum(emotion) for i in emotion]
    
class FERPlus_labeled(FERPlus):
    def __init__(self,indexs, transform=None):
        super(FERPlus_labeled, self).__init__(FERPlus_img_folder_path,'train',transform=transform)

        if indexs is not None:
            self.file_paths = np.array(self.file_paths)[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path)
        label = self.labels[idx]
       
        if self.transform is not None:
            image = self.transform(image)
        
        if self.lmk:
            lmk_path = path.replace('Image/aligned','Landmarks')
            lmk_path = lmk_path.replace('.png', '_landmarks.txt')
            lmks = self.get_landmark_(lmk_path)
            return image, lmks, path, label, idx
        else:
            return image, label

class FERPlus_unlabeled(FERPlus_labeled):
    def __init__(self,  indexs, transform=None):
        super(FERPlus_unlabeled, self).__init__(indexs, transform=transform)
        self.labels = np.array([-1 for i in range(len(self.labels))])
        
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

    train_labeled_set, train_unlabeled_set, test_set = get_ferplus( n_labeled, transform_train=transform_train, transform_val=transform_val)

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