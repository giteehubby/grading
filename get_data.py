import torchvision
from torchvision import transforms
from torchvision.transforms.functional import vflip,hflip
from PIL import Image
import os
from typing import Union, Tuple, List
import random
import torch

def try_gpu(i=0):
    if i + 1 >= torch.cuda.device_count():
        return torch.device('cpu')
    return torch.device(f'cuda:{i}')

def get_imgs(path):
    imgs = dict()
    i = 0
    for root, dirs, files in os.walk(path):
        # 过滤出.jpg文件
        for file in files:
            if file.lower().endswith('.jpg'):
                print(f'read {file}')
                image = Image.open(os.path.join(path,file)).convert('RGB')  # 假设图像是RGB格式的
                imgs[file.split('.')[0]] = image
    return imgs

def get_label(label_path:str) -> dict[str,int]:
    import csv
    label = dict()
    with open(label_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            label[row[0]] = int(row[1])
            
            
    return label

def get_data(path,label_path:str='Mess1_annotation_train.csv',is_train:bool=True,augrate:int=4,valid_num:int=60):

    
    X_imgs = get_imgs(path)
    totensor = transforms.Compose([transforms.ToTensor(),])  # 将PIL图像或numpy.ndarray转换为张量，并归一化到[0, 1]
    
    if is_train:

        X_train, y_train = [], []
        X_valid, y_valid = [], []
        cnt = 0

        labels = get_label(label_path)
        keys = [key for key in X_imgs.keys()]
        random.shuffle(keys)
        for key in keys[:-valid_num]:
            ximage = X_imgs[key]
            lb = labels[key]
            
            for _ in range(augrate):

                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    # 如果需要的话，还可以添加Normalize变换
                ])
                X_train.append(transform(ximage))
                y_train.append(lb)
                cnt = cnt + 1
                print(f'make an augment to {cnt} samples')
            
        for key in keys[-valid_num:]:
            ximage = X_imgs[key]
            lb = labels[key]
            X_valid.append(totensor(ximage))
            y_valid.append(lb)
        return [torch.stack(X_train,dim=0),torch.tensor(y_train,dtype=torch.long),
                torch.stack(X_valid,dim=0),torch.tensor(y_valid,dtype=torch.long)]
    else:
        X = []
        for key in X_imgs.keys():
            ximage = X_imgs[key]
            X.append(transforms.ToTensor()(ximage))
        return torch.stack(X,dim=0),[key for key in X_imgs.keys()]
