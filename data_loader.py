import torch.utils.data as data
from PIL import Image
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import numpy as np
import scipy.io as sio
from torch.autograd import Variable

def MAT_data_process(dataset_path):

    '''处理mat数据，返回[类别, 特征]的list'''

    mat_data_root = os.path.join(dataset_path)
    data = sio.loadmat(mat_data_root)

    prevate_feature_data = []
    num_classes = len(data['X'][0])
    max_len = 0
    for num in range(len(data['X'][0])):
        for lennum in range(len(data['X'][0][num])):
            if len(data['X'][0][num][lennum]) >= max_len:
                max_len = len(data['X'][0][num][lennum])
    print(max_len)
    for i in range(len(data['X'][0][0])):  # 2000
        datalist = []
        for feature in range(len(data['X'][0])):  # 3(class_label)
            data_array = data['X'][0][feature][i]
            #data_array = np.asanyarray(data_array, dtype="float")
            if len(data_array) < max_len:
                data_array = np.pad(data_array, (0, max_len - len(data_array)), 'constant')
            datalist.append([data_array, int(data['Y'][i])])

        prevate_feature_data.append(datalist)
    # print(prevate_feature_data[0][1])
    # exit()
    torch.save(prevate_feature_data, './result/feature_list.pt')
    print('save finish!')


# def MAT_data_process(dataset_path):
#
#     '''处理mat数据，返回[类别, 特征]的list'''
#
#     mat_data_root = os.path.join(dataset_path)
#     data = sio.loadmat(mat_data_root)
#
#     prevate_feature_data = []
#     num_classes = len(data['X'][0])
#     max_len = 0
#     for num in range(len(data['X'][0])):
#         for lennum in range(len(data['X'][0][num])):
#             if len(data['X'][0][num][lennum]) >= max_len:
#                 max_len = len(data['X'][0][num][lennum])
#     datalist = []
#     for i in range(len(data['X'][0][0])):  # 2000
#
#         for feature in range(len(data['X'][0])):  # 3(class_label)
#             data_array = data['X'][0][feature][i]
#             data_array = np.asanyarray(data_array, dtype="uint8")
#             if len(data_array) < max_len:
#                 data_array = np.pad(data_array, (0, max_len - len(data_array)), 'wrap')
#             datalist.append([data_array, int(data['Y'][i])])
#
#     torch.save(datalist, './result/feature_list.pt')
#
#     # print(prevate_feature_data[0][1])
#     # exit()
#     # dataset = ACGetLoader(data_list=prevate_feature_data)
#
#     # return dataset, num_classes, max_len

def Self_data_process(dataset_path):

    '''处理数据，返回[类别, 特征]的list'''
    data = np.load(dataset_path,allow_pickle=True)
#     data2 = np.load('./1000_2_ILSVRC2012_val_feature_file.npy',allow_pickle=True)

    num_classes = 0
    max_len = 0
#     data = np.vstack((data1,data2))
    
    for num in range(len(data)):
        if data[num][1] >= num_classes:
            num_classes = data[num][1]
        if len(data[num][0]) >= max_len:
            max_len = len(data[num][0])

    for i in range(len(data)):
        data[i][0] = np.asanyarray(data[i][0], dtype="float32")
        data[i][1] = data[i][1]
        if len(data[i][0]) <= max_len:
            data[i][0] = np.pad(data[i][0], (0, max_len - len(data[i][0])), 'wrap')

    dataset = ACGetLoader_1(data_list=data)

    return dataset, num_classes, max_len

def AC_mat_data_process(dataset_path):

    '''处理mat数据，返回[类别, 特征]的list'''

    mat_data_root = os.path.join(dataset_path)
    data = sio.loadmat(mat_data_root)
    mat_data_root1 = os.path.join('./1000_2_ILSVRC2012_val_feature_file.npy')
    data1 = sio.loadmat(mat_data_root1)

    prevate_feature_data = []
    num_classes = len(data['X'][0])
    max_len = 0
    for num in range(len(data['X'][0])):
        for lennum in range(len(data['X'][0][num])):
            if len(data['X'][0][num][lennum]) >= max_len:
                max_len = len(data['X'][0][num][lennum])

    for i in range(len(data['X'][0][0])):  # 2000
        datalist = []
        for feature in range(len(data['X'][0])):  # 3(class_label)
            data_array = data['X'][0][feature][i]
            data_array = np.asanyarray(data_array, dtype="float64")
            if len(data_array) < max_len:
                data_array = np.pad(data_array, (0, max_len - len(data_array)), 'wrap')
            datalist.append([data_array, feature])
            
    for i1 in range(len(data1['X'][0][0])):  # 2000
        datalist1 = []
        for feature1 in range(len(data1['X'][0])):  # 3(class_label)
            data_array1 = data1['X'][0][feature1][i1]
            data_array1 = np.asanyarray(data_array1, dtype="float64")
            if len(data_array1) < max_len:
                data_array1 = np.pad(data_array1, (0, max_len - len(data_array1)), 'wrap')
            datalist1.append([data_array1, feature1])

        prevate_feature_data.append([datalist, int(data['Y'][i])])
        prevate_feature_data.append([datalist1, int(data1['Y'][i1])])
    # print(prevate_feature_data[0][1])
    # exit()

    dataset = ACGetLoader(data_list=prevate_feature_data)
#     print(len(dataset))

    return dataset, num_classes, max_len

def AC_mat_data_process_test(dataset_path,matdata_max_len):

    '''处理mat数据，返回[类别, 特征]的list'''

    mat_data_root = os.path.join(dataset_path)
    data = sio.loadmat(mat_data_root)

    prevate_feature_data = []
    num_classes = len(data['X'][0])

    for i in range(len(data['X'][0][0])):  # 2000
        datalist = []
        for feature in range(len(data['X'][0])):  # 3(class_label)
            data_array = data['X'][0][feature][i]
            data_array = np.asanyarray(data_array, dtype="float64")
            if len(data_array) < matdata_max_len:
                data_array = np.pad(data_array, (0, matdata_max_len - len(data_array)), 'wrap')
            datalist.append([data_array, feature])

        prevate_feature_data.append([datalist, int(data['Y'][i])])
    # print(prevate_feature_data[0][1])
    # exit()

    dataset = ACGetLoader(data_list=prevate_feature_data)

    return dataset, num_classes, matdata_max_len

def mat_data_process(dataset_path):

    '''处理mat数据，返回[类别, 特征]的list'''

    mat_data_root = os.path.join(dataset_path)
    data = sio.loadmat(mat_data_root)

    prevate_feature_data = []
    num_classes = len(data['X'][0])
    max_len = 0
    for feature in range(len(data['X'][0])):
        # print(feature)
        for i in range(len(data['X'][0][0])):
            label_cell = len(data['X'][0][feature][i])
            data_array = data['X'][0][feature][i]
            data_array = np.asanyarray(data_array, dtype="uint8")
            if len(data_array) > max_len:
                max_len = len(data['X'][0][feature][i])
            #data_array = np.pad(data_array, (0, max_len - len(data_array)), 'constant')
            data_array = np.pad(data_array, (0, max_len - len(data_array)), 'wrap')
            prevate_feature_data.append([data_array, feature])

    D_dataset = GetLoader(
        data_list=prevate_feature_data)

    return D_dataset, num_classes, max_len

def data_transform(image_size):

    '''训练集数据批量处理'''

    img_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

def data_process(dataset_path, config, Train=True):

    '''处理数据集'''

    img_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    if config.dataset_name == 'MNIST':

        MNIST_dataset = datasets.MNIST(
            root=dataset_path,
            train=Train,
            transform=img_transform,
            download=True)

        return MNIST_dataset

    elif config.dataset_name == ' ':
        print('error')
        exit()

class GetLoader(data.Dataset):
    def __init__(self, data_list, transform=None):

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labers =[]

        for data in data_list:
            self.img_paths.append(data[0])
            self.img_labers.append(data[1])

    def __getitem__(self, item):
        imgs = self.img_paths[item]
        # labels = int(labels)
        imgs, labels = self.img_paths[item], self.img_labels[item]
        labels = int(labels) - 1
        # imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        # if self.transform is not None:
            # print(imgs.type)
            # imgs = self.transform(imgs)
            # labels = int(labels)
        return imgs,labels

    def __len__(self):
        return self.n_data

# class MATGetLoader(data.Dataset):
#
#     def __init__(self, data_list, transform=None):
#         self.n_data = len(data_list)
#
#         self.img_paths = []
#         self.img_cluster_labels = []
#
#         for data in data_list:
#             self.img_paths.append(data[0])
#             self.img_cluster_labels.append(data[1])

class ACGetLoader(data.Dataset):
    def __init__(self, data_list, transform=None):

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_cluster_labels = []

        for data in data_list:
            self.img_paths.append(data[0])
            self.img_cluster_labels.append(data[1])

    def __getitem__(self, item):
        imgs = self.img_paths[item]
        imgs, labels = self.img_paths[item], self.img_cluster_labels[item]
        labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

class ACGetLoader(data.Dataset):
    def __init__(self, data_list, transform=None):

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_cluster_labels = []

        for data in data_list:
            self.img_paths.append(data[0])
            self.img_cluster_labels.append(data[1])

    def __getitem__(self, item):
        imgs = self.img_paths[item]
        
        imgs, labels = self.img_paths[item], self.img_cluster_labels[item]
        labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

class ACGetLoader_1(data.Dataset):
    def __init__(self, data_list, transform=None):

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_cluster_labels = []

        for data in data_list:
            self.img_paths.append(data[0])
            self.img_cluster_labels.append(data[1])

    def __getitem__(self, item):
        imgs = self.img_paths[item]
        labels = self.img_cluster_labels[item]
        labels = int(labels)
#         print(imgs.shape,labels)

        return imgs, labels

    def __len__(self):
        return self.n_data
if __name__ == '__main__':
    # AC_mat_data_process('data/mat_data/MNIST.mat')
    MAT_data_process('data/mat_data/HW.mat')











