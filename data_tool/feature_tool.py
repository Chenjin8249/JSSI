import numpy as np
import cv2 as cv
import os
import torch as torch
import torchvision.models as models
from matplotlib import pyplot as plt
from skimage import feature as ft
from skimage.feature import local_binary_pattern
from skimage import data, filters, io, color
from scipy.ndimage import filters
import sys

sys.path.append("./img_gist_feature/")

from utils_gist import *

def Print(v):
    # print(v.shape)
    # if len(v) > 3000:
    #     print(222)
    return 0

def SIFT_feature(img):
    # print(img.shape)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    features_vector = des.reshape(des.shape[0] * des.shape[1])
    ker = (1.0 / 200.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::200]
    Print(features_vector)
    return features_vector

def BRIEF_feature(img):
    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img,None)
    kp, des = brief.compute(img, kp)
    features_vector = des.reshape(des.shape[0] * des.shape[1])
    # ker = (1.0 / 60.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    # features_vector = filters.convolve1d(features_vector, ker)[::60]
    Print(features_vector)
    return features_vector

def SURF_feature(img):
    surf = cv.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img, None)
    print(des)
    features_vector = des.reshape(des.shape[0] * des.shape[1])
    ker = (1.0 / 5.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::5]
    Print(features_vector)
    return features_vector

def ORB_feature(img):
    orb = cv.ORB_create()
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    print(des)
    features_vector = des.reshape(des.shape[0] * des.shape[1])
    # ker = (1.0 / 10.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    # features_vector = filters.convolve1d(features_vector, ker)[::10]
    Print(features_vector)
    return features_vector

def Canny_feature(img):
    edges = cv.Canny(img, 100, 200)
    features_vector = edges.reshape(edges.shape[0] * edges.shape[1])
    ker = (1.0 / 100.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::100]
    Print(features_vector)
    return features_vector

def RETR_LIST_feature(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 127, 255, 0)
    binary, contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, 2)
    features_vector = hierarchy.reshape(1,-1)
    features_vector = features_vector.squeeze()
    ker = (1.0 / 4.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::4]
    Print(features_vector)
    return features_vector

def RETR_EXTERNAL_feature(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 127, 255, 0)
    binary, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, 2)
    features_vector = hierarchy.reshape(1,-1)
    features_vector = features_vector.squeeze()
    # ker = (1.0 / 2.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    # features_vector = filters.convolve1d(features_vector, ker)[::20]
    Print(features_vector)
    return features_vector

def RETR_CCOMP_feature(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 127, 255, 0)
    binary, contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, 2)
    features_vector = hierarchy.reshape(1,-1)
    features_vector = features_vector.squeeze()
    # ker = (1.0 / 5.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    # features_vector = filters.convolve1d(features_vector, ker)[::40]
    Print(features_vector)
    return features_vector

def CalcHist_feature(img):
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    features_vector = hist.reshape(hist.shape[0] * hist.shape[1])
    # ker = (1.0 / 2.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    # features_vector = filters.convolve1d(features_vector, ker)[::2]
    Print(features_vector)
    return features_vector

def hist2D_feature(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    features_vector = hist.reshape(hist.shape[0] * hist.shape[1])
    ker = (1.0 / 30.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::30]
    Print(features_vector)
    return features_vector

def histROI_feature(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    roihist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([hsv], [0, 1], roihist, [0, 180, 0, 256], 1)
    features_vector = dst.reshape(dst.shape[0] * dst.shape[1])
    ker = (1.0 / 5.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::5]
    Print(features_vector)
    return features_vector

def HOG_feature(img):       #方向梯度直方图
    iimg = cv.cvtColor(img, cv.IMREAD_GRAYSCALE)
    features = ft.hog(img,orientations=9,pixels_per_cell=[8,8],cells_per_block=[2,2],visualize=True, feature_vector= True)
    ker = (1.0 / 2.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features[0], ker)[::2]
    Print(features_vector)
    return features_vector

def Sobel_feature(img):
    iimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = filters.sobel(iimg)
    features_vector = edges.reshape(edges.shape[0] * edges.shape[1])
    ker = (1.0 / 5.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::5]
    Print(features_vector)
    return features_vector

def LBP_feature(img):   #局部二值
    iimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(iimg, 24, 3)
    features_vector = lbp.reshape(lbp.shape[0] * lbp.shape[1])
    ker = (1.0 / 5.0) * np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float)
    features_vector = filters.convolve1d(features_vector, ker)[::5]
    Print(features_vector)
    return features_vector

def Color_moments_feature(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature = np.array([h_mean, s_mean, v_mean, h_std, s_std, v_std, h_thirdMoment, s_thirdMoment, v_thirdMoment])
    Print(color_feature)
    return color_feature

def build_filters():
    filters = []
    ksize = [9, 10, 11, 12, 13]  # gabor尺度，6个
    lamda = np.pi / 6.0  # 波长
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor方向，0°，45°，90°，135°，共四个
        for K in range(5):
            kern = cv.getGaborKernel((ksize[K], ksize[K]), 7.0, theta, lamda, 0.5, 0, ktype=cv.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    Print(filters)
    return filters

def RGB_feature(img):
    img_bgr = cv.cvtColor(img, cv.IMREAD_COLOR)

    # 分别获取三个通道的ndarray数据
    img_b = img_bgr[:, :, 0]
    img_g = img_bgr[:, :, 1]
    img_r = img_bgr[:, :, 2]

    '''按R、G、B三个通道分别计算颜色直方图'''
    b_hist = cv.calcHist([img_bgr], [0], None, [256], [0, 255])
    g_hist = cv.calcHist([img_bgr], [1], None, [256], [0, 255])
    r_hist = cv.calcHist([img_bgr], [2], None, [256], [0, 255])
    m, dev = cv.meanStdDev(img_bgr)  # 计算G、B、R三通道的均值和方差
    # img_r_mean=np.mean(r_hist)  #计算R通道的均值
    b_hist = np.squeeze(b_hist)
    g_hist = np.squeeze(g_hist)
    r_hist = np.squeeze(r_hist)
    Print(b_hist)
    Print(g_hist)
    Print(r_hist)
    return b_hist,g_hist,r_hist

def HSV_feature(img):
    img_bgr = cv.cvtColor(img, cv.IMREAD_COLOR)
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # 分别获取三个通道的ndarray数据
    img_h = img_hsv[:, :, 0]
    img_s = img_hsv[:, :, 1]
    img_v = img_hsv[:, :, 2]

    '''按H、S、V三个通道分别计算颜色直方图'''
    h_hist = cv.calcHist([img_hsv], [0], None, [256], [0, 255])
    s_hist = cv.calcHist([img_hsv], [1], None, [256], [0, 255])
    v_hist = cv.calcHist([img_hsv], [2], None, [256], [0, 255])
    # m,dev = cv2.meanStdDev(img_hsv)  #计算H、V、S三通道的均值和方差
    h_hist = np.squeeze(h_hist)
    s_hist = np.squeeze(s_hist)
    v_hist = np.squeeze(v_hist)
    Print(h_hist)
    Print(s_hist)
    Print(v_hist)
    return h_hist, s_hist, v_hist


def Gabor_feature(img):
    # img = io.imread(img)  # 读取图像
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # RGB转灰度
    frequency = 0.6
    # 调用gabor函数
    real, imag = filters.gabor(img_gray, frequency=0.6, theta=45, n_stds=5)
    # 取模图像
    img_mod = np.sqrt(real.astype(float) ** 2 + imag.astype(float) ** 2)
    # 图像显示
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(img_mod, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(real, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(imag, cmap='gray')
    plt.show()

