# import numpy as np
# import cv2 as cv
# import os
# import torch as torch
# import torchvision.models as models
# from matplotlib import pyplot as plt
# from skimage import feature as ft
# from skimage.feature import local_binary_pattern
# from skimage import data, filters
from feature_tool import *
from deep_tool import *
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
from torchvision import transforms as T

def Data(list):
    f_list = []
    for num in range(len(list)):
        f_list.append([list[num], num])
    return f_list


if __name__ == "__main__":
    total_feature_list = []
    i = 0
    for filename in os.listdir("imagenet_test3"):
        i = i + 1
        print(i, filename)
        imgpath = os.path.join("imagenet_test3" + '/' + filename)
        # imgpath = os.path.join("imagenet_test2" + '/' + 'ILSVRC2012_val_0000063.JPEG')
        img = cv.imread(imgpath)
        img1 = cv.resize(img,(80,80))
        # img = img.resize(224, 224, 3)
        feature_list = []

        # feature
        feature_list.append(SIFT_feature(img1))              # SIFT  0
        # feature_list.append(BRIEF_feature(img1))             # BRIEF 1
        # feature_list.append(SURF_feature(img1))              # SURF  2
        # feature_list.append(ORB_feature(img1))               # ORB   3
        feature_list.append(Canny_feature(img1))             # Canny 4
        # feature_list.append(RETR_LIST_feature(img1))         # RETR_LIST 5
        feature_list.append(RETR_EXTERNAL_feature(img1))     # RETR_EXTERNAL 6
        feature_list.append(RETR_CCOMP_feature(img1))        # RETR_CCOMP    7
        feature_list.append(CalcHist_feature(img1))          # 灰度直方图 8
        feature_list.append(hist2D_feature(img1))            # 2D直方图 9
        feature_list.append(histROI_feature(img1))           # ROI直方图    10
        feature_list.append(HOG_feature(img1))               # 方向梯度*直方图   11
        feature_list.append(Color_moments_feature(img1))     # 颜色矩   12
        feature_list.append(Sobel_feature(img1))             # Sobel算子   13
        feature_list.append(LBP_feature(img1))               # LBP   14

        b_hist, g_hist, r_hist=RGB_feature(img)               # RGB
        feature_list.append(b_hist)
        feature_list.append(g_hist)
        feature_list.append(r_hist)

        h_hist, s_hist, v_hist=HSV_feature(img)               # HSV
        feature_list.append(h_hist)
        feature_list.append(s_hist)
        feature_list.append(v_hist)

        # feature_list.append(Gabor_feature(Gabor_feature))

        # deep feature
        # feature_list.extend(VGG16_feature(imgpath))             # VGG16   15
        # feature_list.extend(VGG16_2(imgpath))
        # feature_list.extend(VGG16_7(imgpath))
        # feature_list.extend(VGG16_12(imgpath))
        feature_list.extend(VGG_feature_total(imgpath))

        output_feature_list = Data(feature_list)

        total_feature_list.extend(output_feature_list)

        # np.save('./feature_file/' + filename + '.npy',total_feature_list)

    # torch.save(total_feature_list, './feature_file/' + filename + '.pt')
    np.save('./500_1_ILSVRC2012_val_feature_file.npy', total_feature_list)