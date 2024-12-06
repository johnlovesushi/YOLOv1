import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F



class YOLOv1_resnet50(nn.Module):
    def __init__(self,num_bboxes = 2,num_classes = 20,num_grids = 7, batch_norm=True):
        super(YOLOv1_resnet50, self).__init__()
        self.num_grids = num_grids
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        self.conv_layers = self._make_conv_layers(batch_norm)
        self.fc_layers = self._make_fc_layers()
        # Initialize the ResNet-50 backbone and keep all layers up to the last convolutional block
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet18backbone = nn.Sequential(*list(resnet.children())[:-2])  # Excludes avgpool and fc layers
    
        self.yolov1head = nn.Sequential(self.conv_layers, self.fc_layers)

        self._initial_weight()

    def forward(self,x):
        S,B,C = self.num_grids,self.num_bboxes,self.num_classes
        output = self.resnet18backbone(x)
        output = self.yolov1head(output)
        return output.view(-1,S,S,C + 5*B)
    
    def _make_conv_layers(self,batch_norm):
        layers = []
        temp_layers = [
            # last ResNet 50 layer consists of a (1x1, 2048) conv layer, and thus we adjust it to 2048
            nn.Conv2d(2048,1024,kernel_size=3,padding = 1),
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
            nn.Linear(S*S*1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + 5 * B )),
            #nn.Sigmoid()
        ])
    
    def _initial_weight(self):
        # only initialize the weight on yolov1head
        for i in range(len(self.yolov1head)):
            if isinstance(self.yolov1head[i],nn.Conv2d):
                nn.init.kaiming_normal_(self.yolov1head[i], mode='fan_in', nonlinearity='leaky_relu')
                if self.yolov1head[i].bias is not None:
                    nn.init.constant_(self.yolov1head[i].bias, 0)
            elif isinstance(self.yolov1head[i],nn.BatchNorm2d):
                nn.init.constant_(self.yolov1head[i].weight, 1)
                nn.init.constant_(self.yolov1head[i].bias, 0)
            elif isinstance(self.yolov1head[i],nn.Linear):
                nn.init.normal_(self.yolov1head[i].weight, 0, 0.01)
                nn.init.constant_(self.yolov1head[i].bias, 0)

def test():
    from torchsummary import summary
    model = YOLOv1_resnet50()
    summary(model, input_size=(3, 448, 448))
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).shape
    print(xshape)
    return 
if __name__ == '__main__':
    test()