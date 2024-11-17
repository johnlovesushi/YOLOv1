# library
import torch
import torchvision.transforms as transforms
from models.yolov1net_resnet18 import YOLOv1_resnet18
import os
import cv2
import argparse
from torch.utils.data import Dataset,  DataLoader, Subset
from voc import VOCDataset
import torchvision.transforms as T
from utils import get_bboxes, convert_bboxes_to_list, plot_boxes,load_class_array
# Load model


# Load image

# Detect objects




if __name__ == '__main__':
    # Paths to input/output images.
    torch.set_printoptions(threshold=torch.inf)
    parser = argparse.ArgumentParser(description='YOLOv1 implementation using PyTorch')
    parser.add_argument('--weight', default='weights/final.pth', help='Model path')
    parser.add_argument('--in_path', default='input/000005.jpg', help='Input image path')
    parser.add_argument('--out_path', default='result.jpg', help='Output imag e path')

    args = parser.parse_args()
    print(args.in_path)
    # Load image.
    image = cv2.imread(args.in_path)
    
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms
            
        def __call__(self, img, bboxes):
            for t in self.transforms:
                img, bboxes = t(img), bboxes
            return img, bboxes

    train_transform = Compose([T.Resize((448, 448)),
                #T.ColorJitter(brightness=[0,1.5], saturation=[0,1.5]),
                T.ToTensor()])
    test_transform = T.Compose([T.ToTensor()])
    train_dataset = VOCDataset(is_train = True, transform= train_transform,transform_scale_translate = False)
    #train_dataset = VOCDataset(is_train = True,transform = train_transform, transform_scale_translate = True)
    subset_size = 2
    dataset_size = len(train_dataset)
    subset_indices = range(1,subset_size)
    subset = Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(dataset = subset, batch_size = 1, 
                              num_workers = 0, shuffle = False)
    
    # for batch_id, (data,target,file_name) in enumerate(train_loader):
    #     print("file_name",file_name)
    #     print("target",target.shape)
        
    #     true_bboxes = convert_bboxes_to_list(target)

    #     print(true_bboxes)

    def get_bboxes_single_image(output):
        pass

    classes = load_class_array()
    print(classes)
    for data,target,file_name in subset:
        print("file_name",file_name)
        print("data",data.shape)
        print("target",target)
        print(data.shape)
        plot_boxes(data, target, classes)
        #true_bboxes = convert_bboxes_to_list(target)
        #print(true_bboxes)
    model = YOLOv1_resnet18()

    #pred_boxes, true_boxes = get_bboxes(train_loader,model)

    #print("true_boxes",true_boxes)
    #print("pred_boxes", pred_boxes)
    #model = YOLOv1_resnet18().to(device)
    #pred_boxes, true_boxes = get_bboxes(train_loader,model)

