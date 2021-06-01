# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:17:02 2021

@author: 19688
"""

import torchvision
from torch.utils.data import DataLoader
from torch import nn


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

#数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

# 使用DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


#%% 搭建神经网络 (为了规范，model一般是单独在一个py文件里)
from model import *
nn_model = Network()

#%% 创建损失函数
loss_fn = nn.CrossEntropyLoss()

#%% 优化器
learnign_rate = 1e-2
optimizer = torch.optim.SGD(nn_model.parameters(), lr=learnign_rate)

#%% 设置训练网络的一些参数

#记录训练，测试次数
total_train_step = 0
total_test_step = 0

# 训练的轮数
epoch = 10

#开始写训练循环:
for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))
    
    # 开始完整的过一遍训练集
    nn_model.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = nn_model(imgs)
        loss = loss_fn(outputs, targets)
        
        # 之前的梯度清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
        
    # 每次训练完一轮之后，自测试集上进行测试
    #过一遍测试集
    total_test_loss = 0
    total_accuracy = 0
    nn_model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = nn_model(imgs)
            
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            
            accuracy  = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("Accuracy: {}".format(total_accuracy / test_data_size))
    
    #torch.save(nn_model, "model_{}.pth".format(i))
    torch.save(nn_model.state_dict(), "model_{}.pth".format(i))