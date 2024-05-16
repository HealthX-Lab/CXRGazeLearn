#!/usr/bin/env python
# coding: utf-8



import sys

from hzhu_metrics_class import *
from hzhu_metrics_saliency import *
from hzhu_data import *
from hzhu_learn import *
from hzhu_MTL_UNet import *
from hzhu_gen import *

import torch
from torch import nn as nn
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import copy

from SupConLoss import SupConLoss
from models.models import GSGNet_T
from models.DenseEDModel import DenseEDNet

matplotlib.use('Agg')
plt.rcParams['axes.facecolor'] = 'white'

import argparse

if __name__ == '__main__':
    print('torch.get_num_threads()=%d'%torch.get_num_threads())

    lr, patience_reduce_lr = 1e-4, 40
    optimizer_dict = {'optimizer':optim.Adam, 'param':{}, 'name':'Adam'}
    lr_factor = 0.1
    lr_min = 1.0e-8
    epoch_max = 1024
    duration_max = 23.5*60*60 #seconds 10.5hour
    patience_early_stop = patience_reduce_lr*2+3
    batch_size = 64
    
    lg_sigma_image = None
    lg_sigma_class = 0.0
    lg_sigma_contrast = 0.0
    
    down = 5
    blur = 500

    classification_loss = nn.CrossEntropyLoss()
    saliency_pred_loss = nn.KLDivLoss(reduction='batchmean')
    contrastive_loss = SupConLoss(temperature=0.07, contrast_mode='all')

    Metrics = {'class':MetricsHandle_Class, 'saliency':MetricsHandle_Saliency}
    Model = MTL_UNet_preset

    name = 'NET'
    folder_string = 'run'
    qH = QuickHelper(path=os.getcwd()+'/'+folder_string)
    print('New Folder name: %s'%qH.ID)
    print(folder_string)

    data_timer = QuickTimer()
    path = '/media/ziruiqiu/OS/data'
    batch_size = 16
    epoch_max = 10
        
    dataAll = DataMaster(path=path, batch_size=batch_size)
    print('Data Preparing time: %fsec'%data_timer())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net = Model(
        device=device,
        out_dict={'class':3, 'image':1},
        loss_dict={'class':lg_sigma_class, 'image':lg_sigma_image})
    Net.save_params(name='Zirui', path=qH())
    #Net.half()
    # for param in Net.parameters():
    #     print(param.dtype)

    #model = GSGNet_T().to(device)
    # model = GSGNet_T()
    # weight = torch.load("/home/ziruiqiu/MscStudy/MT-UNet/Module/models/salicon.pt",map_location='cpu')
    # model.load_state_dict(weight)
    # model.to(device)


    # model = DenseEDNet()
    # model_dict = model.state_dict()
    # # Before loading pretrained dict
    # first_conv_weights_before = next(model.parameters()).clone()
    # pretrained_dict = torch.load('/home/ziruiqiu/MscStudy/MT-UNet/Module/models/SalED_dense_256_pretrained_on_SALICON.pth')
    # model_dict.update(pretrained_dict)

    # model.load_state_dict(model_dict)
    # first_conv_weights_after = next(model.parameters())
    #     # Compare
    # if not torch.equal(first_conv_weights_before, first_conv_weights_after):
    #     print("Model parameters have been updated with pretrained weights.")
    # else:
    #     print("Model parameters have not changed.")
    # model.to(device)

    # model_dict = Net.state_dict()
    # # Before loading pretrained dict
    # first_conv_weights_before = next(Net.parameters()).clone()
    # pretrained_dict = torch.load('/home/ziruiqiu/MscStudy/MT-UNet/Module/run/_J8Bp0/NET_SEWch/NET.pt')
    # model_dict.update(pretrained_dict)

    # Net.load_state_dict(model_dict)
    # # 
    # # print(Net.model.classifier.in_features)
    # # for k in pretrained_dict.keys():
    # #     print(f"Checking parameter: {k}")  # 确认这个循环是否被执行
    # #     if k in model_dict:
    # #         if torch.equal(pretrained_dict[k], model_dict[k]):
    # #             print(f"Parameter {k} has been loaded correctly.")
    # #         else:
    # #             print(f"Parameter {k} has not been loaded correctly.")
    # #     else:
    # #         print(f"Parameter {k} is not in the model_dict.")

    # #'model.classifier.weight', 'model.classifier.bias'
    # Net.out_classification = classification_head(1984,64, 3,0.25)
    # for name, param in Net.named_parameters():
    #     if 'out_classification' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # Net.to(device)
    #print(model)
    netLearn = NetLearn(
        net=Net,
        dataAll=dataAll,
        criterion={'class':classification_loss, 'saliency':saliency_pred_loss, 'contrast':contrastive_loss},
        optimizer_dict=optimizer_dict,
        lr=lr,
        lr_min=lr_min,
        lr_factor=lr_factor,
        epoch_max=epoch_max,
        duration_max=duration_max,
        patience_reduce_lr=patience_reduce_lr,
        patience_early_stop=patience_early_stop,
        device=device,
        metrics=Metrics,
        name=name,
        path=qH())

    netLearn.train()

    print(netLearn.evaluate())
    #netLearn.remove_saved()
    netLearn.remove_saved_optim()
    netLearn.remove_saved_sched()
    netLearn.save_params(name='', path=qH())
    qH.summary()

