import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import json
import torch.nn as nn
from PIL import Image
from skimage import exposure
import pandas as pd
from torchvision.models import resnet18,resnet34, resnet50, resnet101
from hzhu_metrics_class import *
from hzhu_metrics_saliency import *
from hzhu_data import *
from hzhu_learn import *
from hzhu_MTL_UNet import *
from hzhu_gen import *
import torchvision.models as models


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir,file_name):
    print("img.shape: ", img.shape)
    _, H, W= img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # imggray = np.uint8(255 * cam)
    #_, threshold_img = cv2.threshold(src=imggray, thresh=255*0.8, maxval=255 ,type=0)
    # if not os.path.exists(out_dir+'grayscale'):
    #     os.mkdir(out_dir+'grayscale')
    # path_threshold_img = os.path.join(out_dir+'grayscale', file_name)
    # cv2.imwrite(path_threshold_img, np.uint8(255 * cam))
    #print("heatmap.shape: ", heatmap.shape)
    heatmap = np.array(heatmap)
    img = img.cpu().numpy()
    img = np.array(img)
    img = img.transpose(1,2,0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cam_img = 0.3 * heatmap + 0.7 * img * 255
    #cam_img = img * 255 
    # if not os.path.exists(out_dir+'original'):  
    #     os.mkdir(out_dir+'original')
    path_cam_img = os.path.join(out_dir, str(file_name) + '_cam_img.jpg')
    print("path_cam_img: ", path_cam_img)
    cv2.imwrite(path_cam_img, cam_img)

def grad_cam(img,filename,output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    json_path = 'labels.json'

    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value)
               in load_json.items()}
	
	# 只取标签名
    classes = list(classes.get(key) for key in range(3))

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()
    # 定义获取梯度的函数
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def forward_hook(module, input, output):
        fmap_block.append(output)


    net = models.densenet201(pretrained=True)   
    
    net.classifier = nn.Linear(net.classifier.in_features, 3)
    model_dict = net.state_dict()
    pretrained_dict = torch.load('/home/ziruiqiu/MscStudy/MT-UNet/Classfication/log/_jPKRY/NET_DD3dd/NET.pt',map_location=device)
    # 更新当前模型的字典
    pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.to(device)
    net.eval()														# 8
    #print(net)

    # 注册hook
    net.features.denseblock4.register_forward_hook(forward_hook)	# 9
    net.features.denseblock4.register_backward_hook(backward_hook)

    # forward
    img_input = img.unsqueeze(0)
    img_input = img_input.repeat(1,3, 1, 1)
    #img_input = img.unsqueeze(0)
    output = net(img_input)
    predicted_probs = torch.sigmoid(output)
    _, predicted = torch.max(predicted_probs, 1)
    #file.write(str(predicted.cpu().data.numpy()[0][0])+'\n')
    print("predicted: ",predicted.item())
    idx = np.argmax(output.cpu().data.numpy())
    
    print("predict: ",classes[int(predicted.item())]) 
    predict = str(classes[int(predicted.item())])
    #file.write(predict+'\n')
    # backward
    net.zero_grad()
    class_loss = output[0,idx]
    #print("class_loss: ", class_loss)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    # print("grads_val.shape: ", grads_val.shape) # grads_val.shape:  (2048, 7, 7)
    # print("fmap.shape: ", fmap.shape) # fmap.shape:  (2048, 7, 7)
    # 保存cam图片

    
    cam_show_img(img, fmap, grads_val, output_dir,filename)

if __name__ == '__main__':
    path = '/media/ziruiqiu/OS/data'
    batch_size = 6
        
    dataAll = DataMaster(path=path, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    counter = 0
    for data in dataAll('Valid'):
        image = data['cxr'].to(device).unsqueeze(1)
        Y_class = data['Y'].to(device).long()



        for i in range(image.shape[0]):
            grad_cam(image[i],counter,'/home/ziruiqiu/MscStudy/MT-UNet/Module/grad_cam/')
            counter += 1
            
	
	
