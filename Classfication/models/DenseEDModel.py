import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

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


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False, dilated_bool=False):
        super(_DenseLayer, self).__init__()
        if dilated_bool:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                               kernel_size=1, stride=1, bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=3, stride=1, padding=2, dilation=2, bias=False)),
            self.drop_rate = drop_rate
            self.efficient = efficient
        else:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                            kernel_size=1, stride=1, bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)),
            self.drop_rate = drop_rate
            self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride_bool=False):
        super(_Transition, self).__init__()
        if stride_bool:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AdaptiveAvgPool2d((32, 32)))
        else:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False, dilated_bool=False):
        super(_DenseBlock, self).__init__()
        if dilated_bool:
            for i in range(num_layers):

                layer = _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
                    dilated_bool=True
                )
                self.add_module('denselayer%d' % (i + 1), layer)
        else:
            for i in range(num_layers):
                layer = _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
                    dilated_bool=False
                )
                self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseEDNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24), compression=0.5,
                 num_init_features=96, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=False, efficient=False, backbone_grad=True,
                 deconv_fine_grad=True, deconv_coarse_grad=True):

        super(DenseEDNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.dense = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.dense = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.dense.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.dense.add_module('relu0', nn.ReLU(inplace=True))
            self.dense.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
                dilated_bool=False

            )
            self.dense.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression),
                                    stride_bool=False
                                    )

                self.dense.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)


        # Decoding Block
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(2208+1056+128, 848, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(528+384+848+128, 472, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(264+192+192+472+128, 312, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )


        # inception m v2
        self.deconv_inc1 = nn.Sequential(
            nn.Conv2d(312, 96, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),

        )
        self.deconv_inc2 = nn.Sequential(
            nn.ConvTranspose2d(96, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.deconv_inc3 = nn.Sequential(
            # nn.ConvTranspose2d(96, 3, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3)),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.deconv_inc5 = nn.Sequential(
            nn.Conv2d(96, 3, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.deconv_inc6 = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.deconv_inc7 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.deconv_readout = nn.Sequential(
            nn.Conv2d(15, 3, 1, padding=0),
            nn.Conv2d(3, 1, 3, padding=(1, 1)),
            nn.Sigmoid()
        )

        self.scene = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2208, 128, 1),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.trans3_side1 = nn.Sequential(

            nn.ConvTranspose2d(1056, 528, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),

        )
        self.trans3_side2 = nn.Sequential(

            nn.ConvTranspose2d(528, 264, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),

        )

        self.trans2_side1 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        # self.out_classification = classification_head(
        #                 in_features=706560,
        #                 mid_features=1024, out_features=3,
        #                 dropout_rate=0.25)
        
        


    def forward(self, x):
        out = self.dense.conv0(x)
        # assert out.size() == (1, 96, 128, 128)
        out = self.dense.norm0(out)
        out = self.dense.relu0(out)
        out_pre = self.dense.pool0(out)
        # assert out.size() == (1, 96, 64, 64)
        block1 = self.dense.denseblock1(out_pre)
        trans1 = self.dense.transition1(block1)
        # assert trans1.size() == (1, 192, 32, 32)
        out = self.dense.denseblock2(trans1)
        trans2 = self.dense.transition2(out)
        # assert trans2.size() == (1, 384, 16, 16)
        out = self.dense.denseblock3(trans2)
        # assert out.size() == (1, 2112, 16, 16)
        trans3 = self.dense.transition3(out)
        # assert trans3.size() == (1, 1056, 8, 8)
        out = self.dense.denseblock4(trans3)
        # out_flattened = out.view(out.size(0), -1)
        # #print(out_flattened.shape)
        # class_pred = self.out_classification(out_flattened)
        #print(out_pred.shape)
        scene = self.scene(out)
        scene = nn.functional.interpolate(scene, size=(20, 16), mode='bilinear', align_corners=False)
        #print(scene.shape)
        scene_side1 = self.upsample(scene)
        scene_side2 = self.upsample(scene_side1)
        trans3_side1 = self.trans3_side1(trans3)
        trans3_side2 = self.trans3_side2(trans3_side1)
        # trans3_side3 = self.trans3_side3(trans3_side2)
        trans2_side1 = self.trans2_side1(trans2)
        # trans2_side2 = self.trans2_side2(trans2_side1)
        # trans1_side = self.trans1_side(trans1)


        #print(trans3.shape)
        #print(out.shape)
        #print(scene.shape)
        out = torch.cat((trans3, out, scene), 1)
        out = self.deconv_layer1(out)
        # assert out.size() == (1, 816, 16, 16)
        out = torch.cat((trans3_side1, trans2, out, scene_side1), 1)
        out = self.deconv_layer2(out)
        # assert out.size() == (1, 432, 32, 32)
        out = torch.cat((trans3_side2, trans2_side1, trans1, out, scene_side2), 1)
        out = self.deconv_layer3(out)
        # assert out.size() == (1, 270, 64, 64)
        # out = torch.cat((trans3_side3, trans2_side2, trans1_side, out), 1)
        # out = self.deconv_readout(out)
        # assert out.size() == (1, 1, 256, 256)

        out = self.deconv_inc1(out)
        inc2 = self.deconv_inc2(out)
        inc3 = self.deconv_inc3(out)
        # inc4 = self.deconv_inc4(out)
        inc5 = self.deconv_inc5(out)
        inc6 = self.deconv_inc6(out)
        inc7 = self.deconv_inc7(out)
        out = torch.cat((inc2, inc3, inc5, inc6, inc7), 1)
        #print(out.shape)

        out = self.deconv_readout(out)

        out = out.squeeze(1)

        return 1, out


# model = DenseEDNet()
# model_dict = model.state_dict()
# pretrained_dict = torch.load('/home/ziruiqiu/MscStudy/MT-UNet/Module/models/SalED_dense_256_pretrained_on_SALICON.pth')


# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# model_dict.update(pretrained_dict)

# model.load_state_dict(model_dict)
# test = np.zeros((1, 3, 640, 512))
# class_pred, out =model(torch.FloatTensor(test))
# print(class_pred)
# print(out.shape)

