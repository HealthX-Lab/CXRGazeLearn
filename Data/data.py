#!/usr/bin/env python
# coding: utf-8



from hzhu_data_raw import *
import os
from hzhu_gen import *
import argparse



if __name__ == '__main__':
    QH = QuickHelper()
    
    print('Running on my laptop')
    gaze_path = '/home/ziruiqiu/MscStudy/eye-gaze-dataset/physionet.org/files/egd-cxr/1.0.0'
    cxr_path = '/home/ziruiqiu/MscStudy/eye-gaze-dataset'
    save_path = os.getcwd()
    fraction = 1.0
    
    print('Data preparation completed')
    print(QH)

    downsample = 10
    blur = 500
    path_str = 'data'

    local_save_path = save_path+'/'+path_str
    create_folder(local_save_path)

    DATA = MasterDataHandle(gaze_path=gaze_path, cxr_path=cxr_path, blur=blur)
    DATA.save_all(root_path=local_save_path, downsample=downsample, fraction=fraction, seed=0)

    print('%s generation completed'%path_str)
    print(QH)

