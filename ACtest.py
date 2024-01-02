import argparse

import numpy as np
import torch.utils.data
from torchvision.utils import save_image
from ACmodel import *
#from ACKmeans import *
from data_loader import *

class PCAProjectNet(nn.Module):

    def forward(self, features):  # features: NCWH
        k = features.size(1)
        x_mean = (features.sum(dim=1) / k).unsqueeze(1)
        features = features - x_mean



        cov = torch.matmul(features, features.t()) / k
        eigval, eigvec = torch.eig(cov, eigenvectors=True)
        first_compo = eigvec[:, 0]


        projected_map = torch.matmul(first_compo.unsqueeze(0), features)

        maxv = projected_map.max()
        minv = projected_map.min()

        projected_map *= (maxv + minv) / torch.abs(maxv + minv)

        return projected_map

def test(config):
    '''获取处理后的测试集训练数据'''

    fin_Generator = os.path.join('result' + '/Gen_epoch_' + str(config.n_epoch) + '.pth')
    fin_Encoder = os.path.join('result' + '/En_epoch_' + str(config.n_epoch) + '.pth')

    mat_data_path = os.path.join('./data/' + config.dataset_name + '.mat')
#     _, num_classes, matdata_max_len = Self_data_process(config.path)
    matdata_max_len = 3893
    test_datas, _, _ = AC_mat_data_process_test(mat_data_path,matdata_max_len)  # 3,30

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_datas,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    '''导入模型'''
    Generator = SFG(input_size=matdata_max_len, output_size=matdata_max_len, c_size=config.channal_size)
    Encoder = En(input_size=matdata_max_len, output_size=matdata_max_len, c_size=config.channal_size)

    '''导入模型权重'''
    Generator.load_state_dict(torch.load(fin_Generator))
    Encoder.load_state_dict(torch.load(fin_Encoder))

    '''配置'''
    if config.cuda:
        Generator = Generator.cuda()
        Encoder = Encoder.cuda()

    FloatTensor = torch.cuda.FloatTensor if config.cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if config.cuda else torch.LongTensor

    feature_list = []

    '''__testing__'''
    print('        Testing Begin ...')
    for i in range(len(test_dataloader)):
        test_data, cluster_labels = next(iter(test_dataloader))

        #share_feature_ten = Variable(FloatTensor(1, matdata_max_len).fill_(0.0), requires_grad=False)
        #prevate_feature_ten = Variable(FloatTensor(1, matdata_max_len).fill_(0.0), requires_grad=False)

        for j in range(len(test_data)):
            test_img, _ = test_data[j]
            test_imgs = test_img.type(torch.cuda.FloatTensor)

            share_feature = Generator(test_imgs)
            prevate_feature = Encoder(test_imgs)

            if j == 0 :
                share_feature_ten = share_feature   ##################################
                prevate_feature_ten = prevate_feature 
            else:
#                 share_feature_ten = torch.add(share_feature_ten,share_feature)  ##################################
                share_feature_ten = torch.cat((share_feature_ten, share_feature),0)
                prevate_feature_ten = torch.cat((prevate_feature_ten, prevate_feature),1)

#         share_feature_ten_out = share_feature_ten/3 ##################################
        pca = PCAProjectNet()
        share_feature_ten_out = pca(share_feature_ten)
        feature_ = torch.cat((share_feature_ten_out, prevate_feature_ten),1)
        
        if i == 0:
            mat_fea = feature_
            mat_lab = cluster_labels
        else:
            mat_fea = torch.cat((mat_fea, feature_), 0)
            mat_lab = torch.cat((mat_lab, cluster_labels), 0)
        
        feature_list.append([[feature_],[cluster_labels]])

    print('        ')

    print('        Testing Finish ...')

    '''存储提取结果，进行聚类'''
    # torch.save(feature_list, './result/MVfeature_list.pt')
    
    fea = mat_fea.cpu().detach().numpy()
    lab = mat_lab.cpu().detach().numpy()
    
    data_ = {"X": fea, "Y": lab}
    sio.savemat('./result/no_L_Adv_' + config.dataset_name + '.mat', data_)        ####################

if __name__ == "__main__":
    '''测试超参'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--image_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--path", type=str, default='2000_coco_feature_file.npy', help="")         #####################
    parser.add_argument("--channal_size", type=int, default=500, help="")

    '''聚类超参'''
    parser.add_argument("--data_path", type=str, default=r'result', help="")
    parser.add_argument("--n_epoch", type=int, default=20, help="training final epoch")
    parser.add_argument("--maxiter", type=int, default=300, help="")
    parser.add_argument("--dataset_name", type=str, default='Youtube', help="")                       #####################
    
    #Caltech101-7 HW MNIST NUS-WIDE ORL Youtube

    test_opt = parser.parse_args()

    '''导入训练权重'''
    # fin_Generator = os.path.join(test_opt.data_path + '/Gen_epoch_' + str(test_opt.fin_epoch) + '.pth')
    # fin_Encoder = os.path.join(test_opt.data_path + '/En_epoch_' + str(test_opt.fin_epoch) + '.pth')
    # fin_Decoder = os.path.join(test_opt.data_path + '/De_epoch_' + str(test_opt.fin_epoch) + '.pth')

    '''__testing__'''
    test(test_opt)

    '''__cluster__'''
    #Kmeans_cluster(test_opt)

    print('done')