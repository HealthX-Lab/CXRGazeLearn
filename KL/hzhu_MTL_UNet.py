from hzhu_net import *

import torch, os, copy
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
import torchvision.models as models
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
    
)
from loss import *
from models.DenseEDModel import DenseEDNet

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv_block(in_ch, out_ch, dropout_rate=0.25):
    # conv = nn.Sequential(
    #     nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
    #     nn.BatchNorm2d(out_ch),
    #     nn.ReLU(inplace=True),
    #     # SEBlock(out_ch),  # 加入SE块
    #     # nn.Dropout(dropout_rate),
    #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
    #     nn.BatchNorm2d(out_ch),
    #     nn.ReLU(inplace=True),
    # )
    conv = nn.Sequential(
        ResidualBlock(in_ch, out_ch),
        SEBlock(out_ch),
        # ResidualBlock(out_ch, out_ch),
        # SEBlock(out_ch),
    )
        
    return conv

def up_conv_block(in_ch, out_ch, dropout_rate=0.25):
    conv = nn.Sequential(
        # SEBlock(out_ch),  # 加入SE块
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
    # conv = nn.Sequential(
    #     ResidualBlock(in_ch, out_ch),
    #     SEBlock(out_ch),
    #     # nn.Dropout(dropout_rate),
    #     # ResidualBlock(out_ch, out_ch),
    #     # SEBlock(out_ch),
    # )
        
    return conv

def up_conv(in_ch, out_ch):
    up = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        # nn.BatchNorm2d(out_ch),
        # nn.ReLU(inplace=True)
    )

    return up


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out, psi
    
def classification_head(in_features, mid_features, out_features, dropout_rate):
        if mid_features is not None:
            r = nn.Sequential()
            r.add_module('linear_1', nn.Linear(in_features=in_features, out_features=mid_features))
            if dropout_rate is not None:
                if dropout_rate>0.0:
                    r.add_module('dropout', nn.Dropout(p=dropout_rate))
            r.add_module('relu_1', nn.ReLU())
            r.add_module('linear_2', nn.Linear(in_features=mid_features, out_features=out_features))
            return r
        else:
            return nn.Linear(in_features=in_features, out_features=out_features)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.conv_shortcut(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add the input x to the output
        out = F.relu(out)
        return out

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        residual = self.conv_shortcut(x)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += residual  # Add the shortcut to the output
        out = F.relu(out)
        return out

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class UNet_Chunk(Module):
    def __init__(self, in_channels, filter_list):
        super().__init__()

        #densenet
        # self.densenet = models.densenet121(pretrained=True).features
        # for param in self.densenet.parameters():
        #     param.requires_grad = True

        self.in_channels = in_channels
        self.filter_list = filter_list

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(self.in_channels, self.filter_list[0])
        self.Conv2 = conv_block(self.filter_list[0], self.filter_list[1])
        self.Conv3 = conv_block(self.filter_list[1], self.filter_list[2])
        self.Conv4 = conv_block(self.filter_list[2], self.filter_list[3])
        self.Conv5 = conv_block(self.filter_list[3], self.filter_list[4])

        # self.Stem = nn.Conv2d(self.in_channels, self.filter_list[0], kernel_size=3, padding=1)

        # self.Res1 = ResidualBlock(self.filter_list[0], self.filter_list[0])
        # self.Down1 = DownsamplingBlock(self.filter_list[0], self.filter_list[1])

        # self.Res2 = ResidualBlock(self.filter_list[1], self.filter_list[1])
        # self.Down2 = DownsamplingBlock(self.filter_list[1], self.filter_list[2])

        # self.Res3 = ResidualBlock(self.filter_list[2], self.filter_list[2])
        # self.Down3 = DownsamplingBlock(self.filter_list[2], self.filter_list[3])

        # self.Res4 = ResidualBlock(self.filter_list[3], self.filter_list[3])
        # self.Down4 = DownsamplingBlock(self.filter_list[3], self.filter_list[4])

        # self.Res5 = ResidualBlock(self.filter_list[4], self.filter_list[4])

        
        
        # self.Up6 = nn.Sequential(
        #     nn.Conv2d(self.filter_list[4], self.filter_list[3], kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(self.filter_list[3]),
        #     nn.ReLU(inplace=True)
        # )

        self.Up5 = up_conv(1920, self.filter_list[3])
        self.Up_conv5 = up_conv_block(self.filter_list[4]+self.filter_list[3], self.filter_list[3])

        self.Up4 = up_conv(self.filter_list[3], self.filter_list[2])
        self.Up_conv4 = up_conv_block(self.filter_list[3]+self.filter_list[2], self.filter_list[2])

        self.Up3 = up_conv(self.filter_list[2], self.filter_list[1])
        self.Up_conv3 = up_conv_block(self.filter_list[2]+self.filter_list[1], self.filter_list[1])

        self.Up2 = up_conv(self.filter_list[1], self.filter_list[0])
        self.Up_conv2 = up_conv_block(self.filter_list[1]+self.filter_list[0], self.filter_list[0])

        self.Up1 = up_conv(self.filter_list[0], self.filter_list[0])
        self.Up_conv1 = up_conv_block(self.filter_list[0]+self.filter_list[0], self.filter_list[0])

        self.conv = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)

        # self.Up = up_conv(self.filter_list[0], self.filter_list[0])
        # self.Up_conv = up_conv_block(self.filter_list[1], self.filter_list[0])

        self.model = models.densenet201(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 3)
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load('/home/ziruiqiu/MscStudy/MT-UNet/Classfication/log/_jPKRY/NET_DD3dd/NET.pt',map_location=self.device)
        # 更新当前模型的字典
        pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.to(self.device)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        def get_features(module, input, output):
            self.denseblock4_output = output
        self.hook = self.model.features.denseblock4.register_forward_hook(get_features)

        def get_norm5(module, input, output):
            self.norm5_output = output
        self.hook = self.model.features.norm5.register_forward_hook(get_norm5)

    def forward(self, x):
        
        # #densenet
        # e = self.Conv1(x)
        # e0 = self.Maxpool1(e)    
        # e0 = self.Conv2(e0)
        x_class = x.repeat(1, 3, 1, 1)  # 3通道输入
        # features = []
        # # conv0/norm0/relu0 层输出64通道特征
        # x = self.densenet.conv0(x)
        # x = self.densenet.norm0(x)
        # x = self.densenet.relu0(x)
        # # pool0 层输出，并不使用，但是必须应用以进行下一步
        # x = self.densenet.pool0(x)
        # features.append(x)  # features[0]


        # # denseblock1 和 transition1 层输出128通道特征
        # x = self.densenet.denseblock1(x)
        # x = self.densenet.transition1(x)
        # features.append(x)  # features[1]

        # # denseblock2 和 transition2 层输出256通道特征
        # x = self.densenet.denseblock2(x)
        # x = self.densenet.transition2(x)
        # features.append(x)  # features[2]

        # # denseblock3 和 transition3 层输出512通道特征
        # x = self.densenet.denseblock3(x)
        # x = self.densenet.transition3(x)
        # features.append(x)  # features[3]

        # # denseblock4 层输出1024通道特征
        # x = self.densenet.denseblock4(x)
        # features.append(x)  # features[4]
        # e5 = features[-1]  # DenseNet最后的输出
        # e4, e3, e2, e1 = features[-2], features[-3], features[-4], features[-5]  # 中间特征

        class_out = self.model(x_class)

        # encoding path
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        ##residual
        # e1 = self.Stem(x)

        # e2 = self.Res1(e1)
        # e2 = self.Down1(e2)

        # e3 = self.Res2(e2)
        # e3 = self.Down2(e3)

        # e4 = self.Res3(e3)
        # e4 = self.Down3(e4)

        # e5 = self.Res4(e4)
        # e5 = self.Down4(e5)
        # e5 = self.Res5(e5)

        #decoding path
        d5 = self.Up5(self.denseblock4_output)
        # print('d5.shape',d5.shape)
        # print('e5.shape',e5.shape)
        d5 = torch.cat((e5, d5), dim=1)
        d5 = self.Up_conv5(d5)


        d4 = self.Up4(d5)
        # print('d4.shape',d4.shape)
        # print('e4.shape',e4.shape)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # print('d3.shape',d3.shape)
        # print('e3.shape',e3.shape)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # print('d2.shape',d2.shape)
        # print('e2.shape',e2.shape)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        # print('d1.shape',d1.shape)
        # print('e1.shape',e1.shape)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        #densenet
        # d1 = self.Up1(d2)
        # d1 = torch.cat((e0, d1), dim=1)
        # d1 = self.Up_conv1(d1)

        # d0 = self.Up(d1)
        # d0 = torch.cat((e, d0), dim=1)
        # d0 = self.Up_conv(d0)


        #return class_out,d1
        return class_out,d1#, d2, e4, e3, d2, d3, d4, d5
    
    
    
class MTL_UNet(UNet_Chunk):
    
    def __init__(self, in_channels, filter_list, out_dict):
        super().__init__(in_channels, filter_list)
        self.out_dict = out_dict
        self.init()
        
    def init(self):
        self.dummy_tensor = nn.Parameter(torch.tensor(0), requires_grad=False)
        
        if self.out_dict is None:
            self.out_conv = nn.Conv2d(self.filter_list[0], 1, kernel_size=1, stride=1, padding=0)
        else:
            if 'class' in self.out_dict:
                if self.out_dict['class']>0:
                    self.out_classification = classification_head(
                        in_features=self.filter_list[-1],
                        mid_features=self.filter_list[0], out_features=self.out_dict['class'],
                        dropout_rate=0.25)
            if 'image' in self.out_dict:
                if self.out_dict['image']>0:
                    self.out_conv_image = conv_bn_acti_drop(
                        in_channels=self.filter_list[0],
                        out_channels=self.filter_list[0],
                        kernel_size=3,
                        activation=nn.ReLU,
                        normalize=nn.BatchNorm2d,
                        padding=1,
                        dropout_rate=0.0,
                        sequential=None)
                    self.out_conv_image.add_module(
                        'conv_last', nn.Conv2d(self.filter_list[0], self.out_dict['image'], kernel_size=1, stride=1, padding=0))
    
    def forward(self, x):
        
        e5,d2 = super().forward(x)#, e4, e3, d2, d3, d4, d5
        
        if self.out_dict is None:
            # y = self.out_conv(d2)
            # return self.dummy_tensor, y
            return
        else:
            r = []
            if 'class' in self.out_dict:
                if self.out_dict['class']>0:
                    average_pool_e5 = e5.mean(dim=(-2,-1))
                    # average_pool_d2 = d2.mean(dim=(-2,-1))
                    # average_pool = torch.cat((average_pool_e5, average_pool_d2), dim=1)
                    # # print('average_pool.shape',average_pool.shape)                  
                    # y_class = self.out_classification(average_pool)
                    #print('y_class.shape',y_class.shape)
                    # y_class = (y_class1 + y_class2 + y_class3) / 3
                    #print('y_class.shape',y_class.shape)
                    #breakpoint()
                    r.append(e5)
                else:
                    r.append(self.dummy_tensor)
            else:
                r.append(self.dummy_tensor)
                
            if 'image' in self.out_dict:
                if self.out_dict['image']>0:
                    y_image = self.out_conv_image(d2)
                    r.append(y_image)
                else:
                    r.append(self.dummy_tensor)
            else:
                r.append(self.dummy_tensor)

            # if 'contrast' in self.out_dict:
            #     if self.out_dict['contrast']>0:
            #         # # 1. 全局平均池化
            #         # features = F.adaptive_avg_pool2d(e5, (1, 1))
            #         # features = features.view(features.size(0), -1)  # 将特征张量变为(4, 1024)
            #         # features = F.normalize(features, dim=1)
            #         # # 2. 特征分割（这里假设你已经有了两个不同的视图，每个视图两个样本）
            #         # #f1, f2 = features.split(8, dim=0)
            #         # # 3. 使用torch.cat沿着新维度合并特征
            #         # #contrast_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            #         # # 现在contrast_features的形状应该是(2, 2, 1024)，可以用作SupConLoss的输入
            #         r.append(e5)
            #     else:
            #         r.append(self.dummy_tensor)

            return tuple(r)
        
class MTL_UNet_preset(MTL_UNet):
    
    def __init__(self, device, out_dict, loss_dict):
        self.device = device
        base = 64 if os.getcwd()[0] == '/' else 2
        super().__init__(in_channels=1, filter_list=[base*(2**i) for i in range(5)], out_dict=out_dict)
        
        self.loss_dict = loss_dict
        self.mt_param_init()
        
        self.to(self.device)
        # model_dict = self.state_dict()
        # pretrained_dict = torch.load('/home/ziruiqiu/MscStudy/MT-UNet/Module/log/train_encoder/NET_PzAoP/NET.pt')
        # # 更新当前模型的字典
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict)
        # self.to(self.device)
        # for name, param in self.named_parameters():
        #     if 'Conv' in name or 'out_classification' in name:
        #         param.requires_grad = False
        
    def mt_param_init(self):
        if 'class' in self.out_dict:
            if self.out_dict['class']>0:
                if self.loss_dict['class'] is not None:
                    self.lg_sigma_class = nn.Parameter(torch.tensor(self.loss_dict['class'], device=self.device, dtype=torch.float32))
                else:
                    self.lg_sigma_class = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    
        if 'image' in self.out_dict:
            if self.out_dict['image']>0:
                if not isinstance(self.loss_dict['image'], (list, tuple)):
                    self.loss_dict['image'] = [self.loss_dict['image'],]
                for item in self.loss_dict['image']:
                    if item is not None:
                        self.lg_sigma_image = nn.Parameter(torch.tensor(item, device=self.device, dtype=torch.float32))
                    else:
                        self.lg_sigma_image = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if 'contrast' in self.out_dict:
            if self.out_dict['contrast']>0:
                if self.loss_dict['contrast'] is not None:
                    self.lg_sigma_contrast = nn.Parameter(torch.tensor(self.loss_dict['contrast'], device=self.device, dtype=torch.float32))
                else:
                    self.lg_sigma_contrast = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                    
    def compute_loss_class(self, y_pred, y_true, loss_function_class):
        
        sigma = torch.exp(self.lg_sigma_class)
        loss_raw_class = loss_function_class(y_pred, y_true) 
        loss_weighted = loss_raw_class/sigma/sigma+torch.log(sigma+1.0) 
         
        return sigma, loss_raw_class, loss_weighted
    
    def compute_loss_image(self, y_pred, y_true, loss_function, idx):
        
        sigma = torch.exp(self.lg_sigma_image)
        loss_raw = loss_function(y_pred, y_true)
        loss_weighted = loss_raw/sigma/sigma/2.0+torch.log(sigma+1.0)
        
        return sigma, loss_raw, loss_weighted
    
    def compute_loss(self, y_class_pred, y_image_pred, y_class_true, y_image_true, loss_class, loss_image_list):
        
        class_sigma, class_loss_raw, class_loss_weighted = self.compute_loss_class(
            y_pred=y_class_pred, y_true=y_class_true, loss_function_class=loss_class)
        
        image_sigma, image_loss_raw, image_loss_weighted = self.compute_loss_image(
            y_pred=y_image_pred, y_true=y_image_true, loss_function=loss_image_list[0], idx=0)
        

        
        loss_sum = image_loss_raw#class_loss_weighted + image_loss_weighted 
        # print('class_loss_weighted', class_loss_weighted)
        # print('image_loss_weighted', image_loss_weighted)
        # print('contrastive_loss_raw', contrastive_loss_raw)
        r = {'loss_sum':loss_sum,
             'class_loss_raw':class_loss_raw,
             'image_loss_raw':image_loss_raw,
             }
            
        return r
    
    def get_status(self):
        r = []
        r.append(torch.exp(self.lg_sigma_class).detach().clone().cpu())
        r.append(torch.exp(self.lg_sigma_image).detach().clone().cpu())
        return r
    
    def get_status_str(self):
        stats = self.get_status()
        r = ''
        for item in stats:
            r += '%.2e '%item
            
        return r
    
    def compute_loss_contrast(self, contrast_features):
        z1, z2 = contrast_features.split(16, dim=0)
        loss = self.infoNCE(z1, z2) / 2 + self.infoNCE(z2, z1) / 2
        # y_true_modified = y_true[::2]
        # loss = contrastive_loss(contrast_features, y_true_modified)
        
        r = {'contrast_loss_raw':loss}
        return r

    def infoNCE(self, nn, p, temperature=0.7):
        nn = torch.nn.functional.normalize(nn, dim=1)
        p = torch.nn.functional.normalize(p, dim=1)
        nn = self.gather_from_all(nn)
        p = self.gather_from_all(p)
        logits = nn @ p.T 
        logits /= temperature
        n = p.shape[0]
        labels = torch.arange(0, n, dtype=torch.long).cuda()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss
    
    def gather_from_all(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Similar to classy_vision.generic.distributed_util.gather_from_all
        except that it does not cut the gradients
        """
        if tensor.ndim == 0:
            # 0 dim tensors cannot be gathered. so unsqueeze
            tensor = tensor.unsqueeze(0)

        if is_distributed_training_run():
            tensor, orig_device = convert_to_distributed_tensor(tensor)
            gathered_tensors = self.GatherLayer.apply(tensor)
            gathered_tensors = [
                convert_to_normal_tensor(_tensor, orig_device)
                for _tensor in gathered_tensors
            ]
        else:
            gathered_tensors = [tensor]
        gathered_tensor = torch.cat(gathered_tensors, 0)
        return gathered_tensor
        
    class GatherLayer(torch.autograd.Function):
        """
        Gather tensors from all workers with support for backward propagation:
        This implementation does not cut the gradients as torch.distributed.all_gather does.
        """

        @staticmethod
        def forward(ctx, x):
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
            return tuple(output)

        @staticmethod
        def backward(ctx, *grads):
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            return all_gradients[dist.get_rank()]
        
    
    