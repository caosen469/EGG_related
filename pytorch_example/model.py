# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:46:01 2021

@author: 19688
"""
from torch import nn
import torch


#%% 搭建神经网络 (为了规范，model一般是单独在一个py文件里)
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequantial(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

# 测试网络正确性，建立一个有batchsize的实例向量没检查输出的size
if __name__ == "__main__":
    model = Network()
    
    test_input = torch.ones((64, 3, 32, 32))
    test_output = model(test_input)
    
    print(test_output.shape)