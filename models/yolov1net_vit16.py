import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,vit_b_16, ViT_B_16_Weights, ResNet18_Weights


class YOLOv1_vitb16(nn.Module):
    def __init__(self,num_bboxes = 2,num_classes = 20,num_grids = 7, batch_norm=True, image_size = 448):
        super(YOLOv1_vitb16, self).__init__()
        self.image_size = image_size
        self.num_grids = num_grids
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        self.conv_layers = self._make_conv_layers(batch_norm)
        self.fc_layers = self._make_fc_layers()
        # Initialize the vgg19bn backbone and keep all layers up to the last convolutional block

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18 = nn.Sequential(*list(resnet.children())[:-2])  # Excludes avgpool and fc layers
    
        self.vitb16_backbone = vit_b_16(image_size = self.image_size,weights=ViT_B_16_Weights.DEFAULT)
        #self.vitb16_backbone = nn.Sequential(*list(self.vitb16.children()))  # Excludes avgpool and fc layers
        
        self.yolov1head = self.fc_layers

        self._initial_weight()


        # self._resize_positional_embeddings()

    # def _resize_positional_embeddings(self):
    #     """
    #     Resize the positional embeddings to match the new patch grid size.
    #     """
    #     # Extract the original positional embeddings
    #     pos_embed = self.vitb16_backbone.encoder.pos_embedding  # Shape: [1, num_patches + 1, hidden_dim]
    #     num_patches_original = pos_embed.size(1) - 1  # Exclude [CLS] token
    #     hidden_dim = pos_embed.size(2)

    #     # Calculate the new grid size
    #     patch_size = self.vitb16_backbone.patch_size
    #     grid_size_new = self.image_size // patch_size

    #     # Resize positional embeddings
    #     pos_embed_new = nn.functional.interpolate(
    #         pos_embed[:, 1:].reshape(1, int(num_patches_original**0.5), int(num_patches_original**0.5), hidden_dim).permute(0, 3, 1, 2),
    #         size=(grid_size_new, grid_size_new),
    #         mode='bilinear',
    #         align_corners=False
    #     ).permute(0, 2, 3, 1).reshape(1, grid_size_new * grid_size_new, hidden_dim)

    #     # Update the positional embeddings
    #     self.vitb16_backbone.encoder.pos_embedding = nn.Parameter(
    #         torch.cat([pos_embed[:, :1], pos_embed_new], dim=1)
    #     )
    
    def forward(self,x):
        S,B,C = self.num_grids,self.num_bboxes,self.num_classes
        output = self.vitb16_backbone(x)
        output = self.yolov1head(output)
        return output.view(-1,S,S,C + 5*B)
    
    def _make_conv_layers(self,batch_norm):
        layers = []
        temp_layers = [
            # last vgg19bn layer consists of a (1x1, 512) conv layer, and thus we adjust it to 512
            nn.Conv2d(768,1024,kernel_size=3,padding = 1),
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
            nn.Linear(1000, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + 5 * B )),
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
    model = YOLOv1_vitb16()
    #summary(model, input_size=(3, 224, 224))
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).size()
    print(xshape)
    return 
if __name__ == '__main__':
    test()