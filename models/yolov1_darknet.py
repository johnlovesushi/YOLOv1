import torch
import torch.nn as nn
import torch.nn.functional as F
from darknet import DarkNet
from torchsummary import summary




class YOLOv1(nn.Module):
    def __init__(self, features,num_bboxes = 2,num_classes = 20,num_grids = 7, batch_norm=True):
        super(YOLOv1, self).__init__()
        self.num_grids = num_grids
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        self.features = features  # features from Darknet
        self.conv_layers = self._make_conv_layers(batch_norm)
        self.fc_layers = self._make_fc_layers()


    def forward(self,x):
        S,B,C = self.num_grids,self.num_bboxes,self.num_classes
        output = self.features(x)
        output = self.conv_layers(output)
        output = self.fc_layers(output)

        return output.view(-1,S,S,C + 5*B)
    
    def _make_conv_layers(self,batch_norm):
        layers = []
        temp_layers = [
            nn.Conv2d(1024,1024,kernel_size=3,padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024,1024,kernel_size=3, stride= 2, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024,1024,kernel_size=3,padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024,1024,kernel_size=3,padding = 1),
            nn.LeakyReLU(0.1),

        ]

        if batch_norm:
            for temp_layer in temp_layers:
                layers.append(temp_layer)
                if isinstance(temp_layer, nn.Conv2d):
                    layers.append(nn.BatchNorm2d(temp_layer.out_channels))
        else:
            layers = temp_layers
        
        return nn.Sequential(*layers)

    def _make_fc_layers(self):
        S,B,C = self.num_grids,self.num_bboxes,self.num_classes
        return nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + 5 * B )),
            nn.Sigmoid()
        ])
    
def test():
    features = DarkNet(batch_norm = True, conv_only=True)
    yolo = YOLOv1(features)
    summary(yolo, input_size=(3, 448, 448))

if __name__ == '__main__':
    test()