import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict


class DarkNet(nn.Module):
    def __init__(self, batch_norm=True, init_weight=True, conv_only = True):
        super(DarkNet, self).__init__()
        self.features = self._make_conv_layers(batch_norm, conv_only = conv_only)
                # Initialize weights
        if init_weight:
            self._initialize_weights()
        

    def forward(self, x):
        output = self.features(x)
        return output 

    def _make_conv_layers(self, batch_norm, conv_only):

        # first 20 conv layers
        layers = []
        temp_layers = [
            #448 * 448 * 3
            ('conv1',nn.Conv2d(3,64,kernel_size = 7, stride = 2,padding = 3)),
            ('leaky_relu1',nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool1',nn.MaxPool2d(2, stride = 2)),

            # 112 * 112 * 64
            ('conv2',nn.Conv2d(64,192,kernel_size = 3,padding = 1)),
            ('leaky_relu2',nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool2',nn.MaxPool2d(2,stride = 2)),

            # 56*56*256
            ('conv3',nn.Conv2d(192,128,kernel_size = 1)),
            ('leaky_relu3',nn.LeakyReLU(0.1, inplace=True)),
            ('conv4',nn.Conv2d(128,256,kernel_size = 3, padding = 1)),
            ('leaky_relu4',nn.LeakyReLU(0.1, inplace=True)),
            ('conv5',nn.Conv2d(256,256,kernel_size = 1)),
            ('leaky_relu5',nn.LeakyReLU(0.1, inplace=True)),
            ('conv6',nn.Conv2d(256,512,kernel_size = 3, padding = 1)),
            ('leaky_relu6',nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool3',nn.MaxPool2d(2, stride = 2)),
            # 28*28*512
            ('conv7',nn.Conv2d(512,256,kernel_size = 1)),
            ('leaky_relu7',nn.LeakyReLU(0.1, inplace=True)),
            ('conv8',nn.Conv2d(256,512,kernel_size = 3, padding = 1)),
            ('leaky_relu8',nn.LeakyReLU(0.1, inplace=True)),
            ('conv9',nn.Conv2d(512,256,kernel_size = 1)),
            ('leaky_relu9',nn.LeakyReLU(0.1, inplace=True)),
            ('conv10',nn.Conv2d(256,512,kernel_size = 3, padding = 1)),
            ('leaky_relu10',nn.LeakyReLU(0.1, inplace=True)),
            ('conv11',nn.Conv2d(512,256,kernel_size = 1)),
            ('leaky_relu11',nn.LeakyReLU(0.1, inplace=True)),
            ('conv12',nn.Conv2d(256,512,kernel_size = 3, padding = 1)),
            ('leaky_relu12',nn.LeakyReLU(0.1, inplace=True)),
            ('conv13',nn.Conv2d(512,256,kernel_size = 1)),
            ('leaky_relu13',nn.LeakyReLU(0.1, inplace=True)),
            ('conv14',nn.Conv2d(256,512,kernel_size = 3, padding = 1)),
            ('leaky_relu14',nn.LeakyReLU(0.1, inplace=True)),
            ('conv15',nn.Conv2d(512,512,kernel_size = 1)),
            ('leaky_relu15',nn.LeakyReLU(0.1, inplace=True)),
            ('conv16',nn.Conv2d(512,1024,kernel_size = 3, padding = 1)),
            ('leaky_relu16',nn.LeakyReLU(0.1, inplace=True)),
            ('maxpool4',nn.MaxPool2d(2, stride = 2)),
            # 14*14*1024
            ('conv17',nn.Conv2d(1024,512,kernel_size = 1)),
            ('leaky_relu17',nn.LeakyReLU(0.1, inplace=True)),
            ('conv18',nn.Conv2d(512,1024, kernel_size = 3, padding = 1)),
            ('leaky_relu18',nn.LeakyReLU(0.1, inplace=True)),
            ('conv19',nn.Conv2d(1024,512,kernel_size = 1)),
            ('leaky_relu19',nn.LeakyReLU(0.1, inplace=True)),
            ('conv20',nn.Conv2d(512,1024, kernel_size = 3, padding = 1)),
            ('leaky_relu20',nn.LeakyReLU(0.1, inplace=True)),
        ]

        if batch_norm:
            for idx, (name,curr_layer) in enumerate(temp_layers):
                layers.append((name,curr_layer))
                if isinstance(curr_layer, nn.Conv2d):
                    layers.append((f'norm{idx+1}',nn.BatchNorm2d(curr_layer.out_channels)))
        else:
            layers = temp_layers
        
        if not conv_only:
            layers += [
                ('avgpool', nn.AvgPool2d(7)),
                ('flatten', nn.Flatten()),
                ('fc', nn.Linear(1024, 1000)),
            ]
        
        return nn.Sequential(OrderedDict(layers))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_fc_layers(self):
        pass

def test():
    darknet = DarkNet(batch_norm = True, conv_only=True)
    #summary(darknet, input_size=(3, 448, 448))
    print("using 224 figures")
    summary(darknet, input_size=(3, 224, 224))
if __name__ == '__main__':
    test()