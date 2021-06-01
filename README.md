# EGG_related

This repo is used to store other algorithm repo may be used in EGG project.

## Common Spatial Pattern algorithm
https://github.com/spolsley/common-spatial-patterns

## Covariate Shift Detection
https://github.com/erlendd/covariate-shift-adaption/blob/master/Supervised%20classification%20by%20covariate%20shift%20adaption.ipynb

https://www.arangodb.com/2020/11/arangoml-part-4-detecting-covariate-shift-in-datasets/

## Power Spectral Density

## Hyperparameter Tuning
### Grid Search

## EEG Preprocessing
https://github.com/mne-tools/mne-python

# 相关知识

## 信号处理
EEG数据处理步骤：https://zhuanlan.zhihu.com/p/142068015

## 机器学习

## 深度学习
图神经网络

ROC曲线，AOC值：https://zhuanlan.zhihu.com/p/31256633

## Pytorch
安装以及教程https://www.bilibili.com/video/BV1hE411t7RN?from=search&seid=15984300105231899247
注意：anaconda一定要安装64位的

### Pytorch 训练框架
https://github.com/caosen469/EGG_related/tree/main/pytorch_example

### 制作自己的数据集

### 迁移学习
#### 修改别人的模型：https://www.bilibili.com/video/BV1hE411t7RN?p=26
vgg_false = torchvision.models.vgg16(pretrained=False) # 只是加载vgg16的结构
vgg_true = torchvision.models.vgg16(pretrained=True) # 加载了结构，也加载了训练的参数

**修改模型**：
首先打印出网络结构，知道某层，或者某好几层构成的某个模块的名字：print(vgg16_true)

## 多模态机器学习 (EGG+ECG)

# 相关软件
## Matlab
### EGGlab
https://zhuanlan.zhihu.com/p/102264694 EGGlab安装使用

https://sccn.ucsd.edu/eeglab/index.php EGGlab 官方网站，下载，tutorial，workshop。

### Eye-EGG
https://blog.csdn.net/craig_cc/article/details/105530118


