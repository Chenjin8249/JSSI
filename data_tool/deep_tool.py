import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.ndimage import filters
# from keras.preprocessing import image
# from scipy.optimize import fmin_l_bfgs_b
# import time
# import argparse
# import matplotlib.image as mpimg # mpimg 用于读取图片
# from keras.applications import vgg19
# from keras import backend as K
# import matplotlib.pyplot as plt
# from keras.preprocessing import image
# import os
# # model = vgg19.VGG19(weights='imagenet', include_top=False,)
# from keras.models import Model


def VGG16_feature(imgpath,list):
    model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    # model.cuda()    # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    model.eval()  # 必须要有，不然会影响特征提取结果

    img = Image.open(imgpath)  # 读取图片
    if img.mode == 'L':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_to_tensor = transforms.ToTensor()
    tensor = img_to_tensor(img)  # 将图片转化成tensor

    # tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉
    tensor = tensor.unsqueeze(dim=0)
    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
    features_vector = result_npy[0].reshape(512, -1)
    # list = []
    for i in range(len(features_vector)):
        list.append(features_vector[i])

    return list

    # return result_npy[0]  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]

def VGG16_2(imgpath,list):
    model = models.vgg16(pretrained=True).features[:2]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    # model.cuda()    # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    model.eval()  # 必须要有，不然会影响特征提取结果

    img = Image.open(imgpath)  # 读取图片
    if img.mode == 'L':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_to_tensor = transforms.ToTensor()
    tensor = img_to_tensor(img)  # 将图片转化成tensor

    # tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉
    tensor = tensor.unsqueeze(dim=0)
    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
    features_vector = result_npy[0].reshape(64, -1)

    # list = []
    for i in range(len(features_vector)):
        ker = (1.0 / 50.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
        features = filters.convolve1d(features_vector[i], ker)[::20]
        # print(len(features))
        list.append(features)

    return list

    # return result_npy[0]  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]

def VGG16_7(imgpath,list):
    model = models.vgg16(pretrained=True).features[:7]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    # model.cuda()    # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    model.eval()  # 必须要有，不然会影响特征提取结果

    img = Image.open(imgpath)  # 读取图片
    if img.mode == 'L':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_to_tensor = transforms.ToTensor()
    tensor = img_to_tensor(img)  # 将图片转化成tensor

    # tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉
    tensor = tensor.unsqueeze(dim=0)
    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错

    features_vector = result_npy[0].reshape(128, -1)

    # list = []
    for i in range(len(features_vector)):
        ker = (1.0 / 10.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
        features = filters.convolve1d(features_vector[i], ker)[::5]
        # print(len(features))
        list.append(features)
        
    return list

    # return result_npy[0]  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]

def VGG16_12(imgpath,list):
    model = models.vgg16(pretrained=True).features[:12]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    # model.cuda()    # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    model.eval()  # 必须要有，不然会影响特征提取结果

    img = Image.open(imgpath)  # 读取图片
    if img.mode == 'L':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_to_tensor = transforms.ToTensor()
    tensor = img_to_tensor(img)  # 将图片转化成tensor

    # tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉
    tensor = tensor.unsqueeze(dim=0)
    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
    features_vector = result_npy[0].reshape(256, -1)

    # list = []
    for i in range(len(features_vector)):
        ker = (1.0 / 2.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
        features = filters.convolve1d(features_vector[i], ker)[::2]
        # print(len(features))
        list.append(features)
    return list

def VGG_feature_total(imgpath):

    list = []

    list1 = VGG16_feature(imgpath, list)
    # print(len(list1))
    list2 = VGG16_2(imgpath, list1)
    # print(len(list2))
    list3 = VGG16_7(imgpath, list2)
    # print(len(list3))
    list4 = VGG16_12(imgpath, list3)
    # print(len(list4))



    
    
    return list4