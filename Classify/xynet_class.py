from __future__ import print_function, division
import os
import torch
import pandas as pd
import torchvision
import torch.nn as nn
import torch.optim as optim
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from PIL import Image
from torchvision.datasets import ImageFolder
#from logger import Logger
from tensorboardX import SummaryWriter
import copy
from xy_Net import xy_Net
#from Detnet import Detnet


os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定一块gpu为可见
writer = SummaryWriter('./log/4channel/detnet')

data_transform = transforms.Compose([
    transforms.RandomRotation(180, resample=False, expand=False),

    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形转
    #transforms.T4channel(),
    transforms.ToTensor(), 
#R_mean is 0.552623, G_mean is 0.560849, B_mean is 0.620024,Grey_mean is 0.565106
#R_var is 0.784412, G_var is 0.789312, B_var is 0.818015,Grey_var is 0.565106
####H1数据集
#R_mean is 0.340893, G_mean is 0.309973, B_mean is 0.375995,Grey_mean is 0.326742
#R_var is 0.605642, G_var is 0.576078, B_var is 0.636346,Grey_var is 0.326742
#R_mean is 0.816100, G_mean is 0.828175, B_mean is 0.848629,Grey_mean is 0.826902
#R_var is 0.923045, G_var is 0.928421, B_var is 0.935280,Grey_var is 0.826902


    # transforms.Normalize(mean=[0.340893, 0.309973, 0.375995,326742],
    #                      std=[0.605642,0.576078, 0.636346,326742])

    transforms.Normalize(mean=[0.465391, 0.490530, 0.537757],
                         std=[0.694476,0.712328, 0.741298])
    #transforms.Normalize(mean=[0.816100, 0.828175, 0.848629],
                         #std=[0.923045,0.928421, 0.935280])
])


train_dir='/data/xy_data/datasets/3_29/classify/train/'
test_dir='/data/xy_data/datasets/3_29/classify/val/'
                                   
train_dataset = ImageFolder(train_dir,transform=data_transform)
test_dataset=ImageFolder(test_dir,transform=data_transform)
print('Num training images: ', len(train_dataset))

batch_size =32
num_workers=8

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)



test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

model = Detnet()

#model = models.resnet101()
renext_model=torch.load('detnet59.pth')
#model_dict= model.state_dict()
#state_dict={k:v for k,v in renext_model.items() if k in model_dict.keys()}
#model_dict.update(state_dict)
class_num = 2  # imagenet的类别数是1000
channel_in = model.fc.in_features  # 获取fc层的输入通道数
#model.fc = nn.Linear(channel_in, class_num)  # 最后一层替换
#model.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
model.fc=nn.Sequential(
     nn.Dropout(0.5),
     nn.Linear(channel_in, 1024),
     nn.ReLU(inplace=True),
     nn.Dropout(0.5),
     nn.Linear(1024, 256),
     nn.ReLU(inplace=True),
     nn.Linear(256, class_num),
         )
#detnet_state=load_state_dict_from_url('https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50-32x4d-0ab1a123.pth')

#detnet_state=torch.load('detnet59.pth')
#model_dict=model.state_dict()
#state_dict={k:v for k,v in detnet_state.items() if k in model_dict }
#print(state_dict.keys())
#model_dict.update(state_dict)
#model.load_state_dict(model_dict)
#weight=model.conv1.weight.clone()
#model.conv1=nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False)
#with torch.no_grad():
#    model.conv1.weight[:,:3]=weight
#    model.conv1.weight[:,3]=model.conv1.weight[:,0]
#num_classes=2
#model.fc=nn.Sequential(
     #nn.Dropout(0.5),
     #nn.Linear(256 * 6* 6, 1024),
     #nn.ReLU(inplace=True),
 #    nn.Dropout(0.5),
  #   nn.Linear(1024, 256),
   #  nn.ReLU(inplace=True),
    # nn.Linear(256, num_classes),
     #    )
for layer in model.fc:
     if isinstance(layer,nn.Linear):
          nn.init.kaiming_normal_(layer.weight.data,nonlinearity='relu')
     if isinstance(layer,nn.Dropout):
          drop=0

     elif isinstance(layer,nn.ReLU):
          relu=0
     else:
          nn.init.kaiming_normal_(layer.weight.data,nonlinearity='relu')
#model.load_state_dict(torch.load('/data/xy_data/code/classify/models/3/xynet423/model_500.pth'))
device = torch.device("cuda:0")
model = model.to(device)
criterion = nn.CrossEntropyLoss()

learning_rate=1e-3
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,momentum=0.9)

def adjust_lr(optimizer,epoch):
	
    # if epoch >=300:
    #     #learning_rate=learning_rate*0.5
    #     lr=learning_rate*0.5

    # elif epoch>=500:
    #     lr=learning_rate*0.1
    # else:
    #     lr=learning_rate   
    lr=learning_rate*(0.1**(epoch//20))
    for param_group in optimizer.param_groups:
        
        param_group['lr']=lr

nums_epoch = 1000  # 为了快速，只训练一个epoch

print('training begin!')
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
best_val=0
for epoch in range(nums_epoch):
    adjust_lr(optimizer,epoch)
    train_loss = 0
    train_acc = 0
    print (optimizer.param_groups[0]['lr'])
    model = model.train()
    print('Epoch ' + str(epoch+1) + ' begin!')
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)

        # 前向传播
        out = model(img)
        optimizer.zero_grad()
        loss = criterion(out, label)
        #print('Train loss in current Epoch' + str(epoch+1) + ':' + str(loss))
        #print('BP begin!')
        # 反向传播
        loss.backward()
        #print('BP done!')
        optimizer.step()

        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        #print('Train accuracy in current Epoch' + str(epoch) + ':' + str(acc))
    epoch_loss=train_loss / len(train_loader)
    epoch_acc=train_acc / len(train_loader)    
    losses.append(epoch_loss)
    acces.append(epoch_acc)
    print('Epoch' + str(epoch+1)  + ' Train  done!')
    print('Epoch' + str(epoch+1)  + ' Test  begin!')
    #epoch1=epoch+1
    # info={'loss':train_loss / len(train_loader),'accuracy':train_acc / len(train_loader)}
    # for tag,value in info.items():
    #     train_logger.scalar_summary(tag,value,epoch+1)
    writer.add_scalar('Train/Loss',epoch_loss,epoch+1)
    writer.add_scalar('Train/Acc',epoch_acc,epoch+1)


    # 每个epoch测一次acc和loss
    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img1, label1 in test_loader:
        img1 = img1.to(device)
        label1 = label1.to(device)
        out = model(img1)

        loss = criterion(out, label1)
        # print('Test loss in current Epoch:' + str(loss))

        # 记录误差
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label1).sum().item()
        acc = num_correct / img1.shape[0]
        eval_acc += acc

    print('Epoch' + str(epoch+1)  + ' Test  done!')
    eval_epoch_loss=eval_loss / len(test_loader)
    eval_epoch_acc=eval_acc / len(test_loader)
    eval_losses.append(eval_epoch_loss)
    eval_acces.append(eval_epoch_acc)
    if eval_epoch_acc>best_val:
        best_val=eval_epoch_acc
        best_models_wgts=copy.deepcopy(model.state_dict())
        best_epoch=epoch+1
    
    print('Epoch {} Train Loss {} Train  Accuracy {} Test Loss {} Test Accuracy {}  '.format(
        epoch + 1, epoch_loss, epoch_acc, eval_epoch_loss,
            eval_epoch_acc))

    writer.add_scalar('Val/Loss',eval_epoch_loss,epoch+1)
    writer.add_scalar('Val/Acc',eval_epoch_acc,epoch+1)


    # info={'loss':eval_loss / len(test_loader),'accuracy':eval_acc / len(test_loader)}
    # for tag,value in info.items():
    #     val_logger.scalar_summary(tag,value,epoch+1)
    model_dir1=os.path.join('/data/xy_data/code/classify/models/3/detnet/model_'+str(epoch+1)+'.pth')
    #torch.save(model, model_dir)
    torch.save(model.state_dict(), model_dir1)
    print('model saved done!')

# print('Best acc:{}'.format(best_epoch))
#     #model_dir=os.path.join('/data/xy_data/code/classify/models/3/2/model_'+str(epoch+1)+'.pkl')
# model_dir1=os.path.join('/data/xy_data/code/classify/models/4channel/3/model_'+str(best_epoch)+'.pth')
#     #torch.save(model, model_dir)
# torch.save(best_models_wgts, model_dir1)
# print('model saved done!')
