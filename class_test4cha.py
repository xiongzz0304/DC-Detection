from __future__ import print_function, division
import os
import torch
import pandas as pd
import torchvision
import torch.nn as nn
import torch.optim as optim
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from torchvision import models
from PIL import Image
from logger import Logger
import cv2
from xy_Net1 import xy_Net
from Detnet import Detnet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定一块gpu为可见
data_transform = transforms.Compose([

    #transforms.RandomRotation(180, resample=False, expand=False),
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    #transforms.RandomHorizontalFlip(0.5),
    #transforms.RandomVerticalFlip(0.5),


    #transforms.T4channel(),
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    #transforms.Normalize(mean=[0.340893, 0.309973, 0.375995,326742],
                         #std=[0.605642,0.576078, 0.636346,326742])
   # transforms.Normalize(mean=[0.43125, 0.455659, 0.50156, 0.453574],
                        # std=[0.692796,0.707797, 0.734721, 0.453574])  # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。,0.488382,0.488382
    transforms.Normalize(mean=[0.465391, 0.490530, 0.537757],
                         std=[0.694476,0.712328, 0.741298])
    # 即：Normalized_image=(image-mean)/std。
    #transforms.Normalize(mean=[0.816100, 0.828175, 0.848629],
     #                    std=[0.923045,0.928421, 0.935280])
])
#val_transform


print('Test data load begin!')
test_dir='/data/xy_data/datasets/3_29/classify/test/'
#test_dir='/data/xy_data/datasets/4_1/H_jpg/test/'
#test_dir='/data/xy_data/datasets/4_23/classify/test/'
test_dataset=ImageFolder(test_dir,transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=1)
print(type(test_data))
print('Test data load done!')

print('load model begin!')
model = Detnet()
#model = models.resnet101()
#model.conv1=nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False)

num_classes=2
channel_in = model.fc.in_features 
#model.fc=nn.Linear(channel_in, num_classes)
model.fc=nn.Sequential(
     nn.Dropout(0.5),
     nn.Linear(channel_in, 1024),
     nn.ReLU(inplace=True),
     nn.Dropout(0.5),
     nn.Linear(1024, 256),
     nn.ReLU(inplace=True),
     nn.Linear(256, num_classes),
         )
#model.load_state_dict(torch.load('models/4/h8/model_140.pth'))
#print(torch.load('models/4/h1/model_16.pth'))
model.load_state_dict(torch.load('/data/xy_data/code/classify/models/3/detnet/model_36.pth'))
#xynet1_state=torch.load('/data/xy_data/code/classify/models/3/xynet1424/model_1000.pth')
#model_dict=model.state_dict()
#state_dict={k:v for k,v in xynet1_state.items() if k in model_dict }
#print(state_dict.keys())
#model_dict.update(state_dict)
#model.load_state_dict(model_dict)
#model =models.resnet50()
#class_num=2
#channel_in = model.fc.in_features  # 获取fc层的输入通道数
#model.fc = nn.Linear(channel_in, class_num)

#pth_from='/data/xy_data/code/classify/models/3/3/model_800.pth'
#model.load_state_dict(torch.load(pth_from))
model.eval()  # 固定batchnorm，dropout等，一定要有
device = torch.device("cuda:0")
model= model.to(device)
print('load model done!')


#测试单个图像属于哪个类别
'''
torch.no_grad()
img = Image.open('/home/momo/mnt/data2/datum/raw/val2/n01440764/ILSVRC2012_val_00026064.JPEG')
img = transform(img).unsqueeze(0)
img_= img.to(device)
outputs = net(img_)
_, predicted = torch.max(outputs,1)
print('this picture maybe:' + str(predicted))
'''
#批量测试准确率,并输出所有测试集的平均准确率
eval_acc = 0
torch.no_grad()
TP=0
FP=0
TN=0
FN=0

for img1, label1 in test_data:
    img1 = img1.to(device)
    label1 = label1.to(device)
    out = model(img1)

    _, pred = out.max(1)
    #print(pred.data)
    #print(label1.data)
    pred=pred.cpu()
    label1=label1.cpu()
    TP+=((pred.numpy()==1)&(label1.numpy()==1)).sum()
    TN+=((pred.numpy()==0)&(label1.numpy()==0)).sum()
    FN+=((pred.numpy()==0)&(label1.numpy()==1)).sum()
    FP+=((pred.numpy()==1)&(label1.numpy()==0)).sum()
    num_correct = (pred == label1).sum().item()
    acc = num_correct / img1.shape[0]
    #print('Test acc in current batch:' + str(acc))
    eval_acc +=acc
print(TP,TN,FP,FN)
print('final acc in Test data:' + str(eval_acc / len(test_data)))
print(type(TP))
p=TP/(TP+FP)
r=TP/(TP+FN)
f1=2*r*p/(r+p)
acc1=(TP+TN)/(TP+TN+FP+FN)
spec=TN/(TN+FP)
sens=TP/(TP+FN)
h_means=2*(spec*sens)/(spec+sens)
print('precision:{} recall:{} f1:{} acc:{} spec:{} sens:{} h_means:{} '.format(p,r,f1,acc1,spec,sens,h_means))





