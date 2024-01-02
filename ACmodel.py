import torch
import torch.nn as nn
from functions import ReverseLayerF
import numpy as np
import torch.nn.functional as F


class SFG(nn.Module):
    def __init__(self, input_size, output_size, c_size):
        super(SFG, self).__init__()
        #########################################
        # Shared Feature Generator
        #########################################
        self.shared_feature_generator = nn.Sequential()
        self.shared_feature_generator.add_module('sd_fc1',nn.Linear(in_features=input_size, out_features=c_size))
        self.shared_feature_generator.add_module('sg_ac1', nn.LeakyReLU(True))
#         self.shared_feature_generator.add_module('sd_fc2',nn.Linear(in_features=c_size, out_features=c_size))
#         self.shared_feature_generator.add_module('sg_ac2', nn.LeakyReLU(True))
#         self.shared_feature_generator.add_module('sd_fc3',nn.Linear(in_features=c_size, out_features=c_size))
#         self.shared_feature_generator.add_module('sg_ac3', nn.LeakyReLU(True))
        self.shared_feature_generator.add_module('sd_fc4',nn.Linear(in_features=c_size, out_features=output_size))
        self.shared_feature_generator.add_module('sg_ac4', nn.LeakyReLU(True))
    def forward(self, input_data):
        input_data = input_data.view(input_data.size(0), -1)
        img = self.shared_feature_generator(input_data)
        return img

class SFD(nn.Module):
    def __init__(self, input_size, output_size, c_size):    #classify
        super(SFD, self).__init__()
        #########################################
        # Shared Feature Discriminator
        #########################################
        self.shared_feature_discriminator = nn.Sequential()
        self.shared_feature_discriminator.add_module('sf_fc1', nn.Linear(in_features=input_size, out_features=c_size))
        self.shared_feature_discriminator.add_module('sf_ac1', nn.LeakyReLU(True))
#         self.shared_feature_discriminator.add_module('sf_fc2', nn.Linear(in_features=c_size, out_features=c_size))
#         self.shared_feature_discriminator.add_module('sf_ac2', nn.LeakyReLU(True))
#         self.shared_feature_discriminator.add_module('sf_fc3', nn.Linear(in_features=c_size, out_features=c_size))
#         self.shared_feature_discriminator.add_module('sf_ac3', nn.LeakyReLU(True))
        self.shared_feature_discriminator.add_module('sf_fc4', nn.Linear(in_features=c_size, out_features=c_size))
        self.shared_feature_discriminator.add_module('sf_ac4', nn.Sigmoid())

        self.adv_layer = nn.Sequential()        # 辨别真假
        self.adv_layer.add_module('avl_fc', nn.Linear(in_features=c_size, out_features=1))
        self.adv_layer.add_module('avl_ac', nn.Sigmoid())

        self.aux_layer = nn.Sequential()        # 分类
        self.aux_layer.add_module('aux_fc', nn.Linear(in_features=c_size, out_features=output_size+1))
        self.aux_layer.add_module('aux_ac', nn.Softmax())

    def forward(self, input_data):
        img_flat = input_data.view(input_data.size(0), -1)
        img = self.shared_feature_discriminator(img_flat)
        validity = self.adv_layer(img)
        label = self.aux_layer(img)
        return validity, label

class En(nn.Module):
    def __init__(self, input_size, output_size, c_size):
        super(En, self).__init__()
        #########################################
        # Private Feature Encoder
        #########################################
        self.private_feature_encoder = nn.Sequential()
        self.private_feature_encoder.add_module('en_fc1', nn.Linear(in_features=input_size, out_features=c_size))
        self.private_feature_encoder.add_module('en_ac1', nn.LeakyReLU(True))
#         self.private_feature_encoder.add_module('en_fc2', nn.Linear(in_features=c_size, out_features=c_size))
#         self.private_feature_encoder.add_module('en_ac2', nn.LeakyReLU(True))
#         self.private_feature_encoder.add_module('en_fc3', nn.Linear(in_features=c_size, out_features=c_size))
#         self.private_feature_encoder.add_module('en_ac3', nn.LeakyReLU(True))
        self.private_feature_encoder.add_module('en_fc4', nn.Linear(in_features=c_size, out_features=output_size))
        self.private_feature_encoder.add_module('en_ac4', nn.LeakyReLU(True))

    def forward(self, input_data):
        input_data = input_data.view(input_data.size(0), -1)
        img = self.private_feature_encoder(input_data)
        return img

class De(nn.Module):
    def __init__(self, input_size, output_size, c_size):
        super(De, self).__init__()
        #########################################
        # Private Feature Encoder
        #########################################
        self.final_decoder = nn.Sequential()
        self.final_decoder.add_module('de_fc1', nn.Linear(in_features=input_size, out_features=c_size))
        self.final_decoder.add_module('de_ac1', nn.LeakyReLU(True))
#         self.final_decoder.add_module('de_fc2', nn.Linear(in_features=c_size, out_features=c_size))
#         self.final_decoder.add_module('de_ac2', nn.LeakyReLU(True))
#         self.final_decoder.add_module('de_fc3', nn.Linear(in_features=c_size, out_features=c_size))
#         self.final_decoder.add_module('de_ac3', nn.LeakyReLU(True))
        self.final_decoder.add_module('de_fc4', nn.Linear(in_features=c_size, out_features=output_size))
        self.final_decoder.add_module('de_ac4', nn.LeakyReLU(True))

    def forward(self, input_data):
        img = self.final_decoder(input_data)
        return img
