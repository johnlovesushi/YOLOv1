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
from utils import get_bboxes, convert_bboxes_to_list, plot_boxes,load_class_array,non_max_suppression
from models.yolov1net_resnet18 import YOLOv1_resnet18
from models.yolov1net_vgg19bn import YOLOv1_vgg19bn
from models.yolov1net_resnet50 import YOLOv1_resnet50
import numpy as np
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = load_class_array()
model_name_dict = {'vgg19bn': lambda: YOLOv1_vgg19bn().to(device),
                   'resnet18bn': lambda: YOLOv1_resnet18().to(device),
                   'resnet50bn': lambda: YOLOv1_resnet50().to(device)
                   }

transform = T.Compose([T.ToTensor()])

# Detect objects
if __name__ == '__main__':
    # Paths to input/output images.
    torch.set_printoptions(threshold=torch.inf)
    parser = argparse.ArgumentParser(description='YOLOv1 implementation using PyTorch')
    parser.add_argument('-m', "--model", dest='model_name', type=str, default='resnet18bn')
    parser.add_argument('--weight', dest='weight_path',default='weights/final.pth', help='Model path')
    parser.add_argument('--in_path', dest='in_path',default='input', help='Input image path')
    parser.add_argument('-f','--file_name',dest = 'file_name', default = '000005.jpg', help='Input image file name')
    parser.add_argument('--out_path', dest='out_path',default='putput', help='Output imag path')

    args = parser.parse_args()
    print("args:", args)

    # load model
    if args.model_name in model_name_dict:
        # Call the corresponding function
        model = model_name_dict[args.model_name]().to(device)
        print(f"Initialized Model: {args.model_name}")
    else:
        raise ValueError(
            f"Invalid model name: {args.model_name}. Choose from {list(model_name_dict.keys())}")

    path_cpt_file = f'cpts/yolov1net_{args.model_name}.cpt'

    # try:
    #     checkpoint = torch.load(path_cpt_file)
    #     model.load_state_dict(checkpoint['model_state_dict'],weights_only=True)
    #     model.eval()
    #     print(f"Petrained {args.model_name} network initalized.")
    # except:
    #     raise ValueError(f"no proper {args.model_name} pretrained model is available ")
    
    #Load image and modify its format
    input_path = os.path.join(args.in_path, args.file_name)
    print("input_path",input_path)
    image_bgr = cv2.imread(input_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    frame = np.array(image)
    frame = cv2.resize(frame, (448, 448))

    # model needs one more dimension as the batch_size
    input_frame = transform(frame).unsqueeze(0).to(device)
    preds = model(input_frame.to(device))
    # squeeze the first batch_size dimension
    data = input_frame.squeeze(0).to(device)
    plot_boxes(data, preds, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=args.file_name, output_folder = "./output/")

#########################################
    # class Compose(object):
    #     def __init__(self, transforms):
    #         self.transforms = transforms
            
    #     def __call__(self, img, bboxes):
    #         for t in self.transforms:
    #             img, bboxes = t(img), bboxes
    #         return img, bboxes


    # train_transform = Compose([T.Resize((448, 448)),
    #             #T.ColorJitter(brightness=[0,1.5], saturation=[0,1.5]),
    #             T.ToTensor()])
    # test_transform = T.Compose([T.ToTensor()])
    # train_dataset = VOCDataset(is_train = True, transform= train_transform,transform_scale_translate = False)
    # #train_dataset = VOCDataset(is_train = True,transform = train_transform, transform_scale_translate = True)
    # subset_size = 10
    # dataset_size = len(train_dataset)
    # subset_indices = range(1,subset_size)
    # subset = Subset(train_dataset, subset_indices)
    
    # train_loader = DataLoader(dataset = subset, batch_size = 1, 
    #                           num_workers = 0, shuffle = False)
    

    # def get_bboxes_single_image(output):
    #     pass

    # classes = load_class_array()
    # print(classes)
    # for data,target in subset:
    #     #print("file_name",file_name)
    #     print("data",data.shape)
    #     print("target",target.shape)
    #     #print(data.shape)
    #     plot_boxes(data, target, classes)
    #     #true_bboxes = convert_bboxes_to_list(target)
    #     #print(true_bboxes)
    # model = YOLOv1_resnet18()

    # pred_boxes, true_boxes = get_bboxes(train_loader,model)

    # print("true_boxes",true_boxes)
    # print("pred_boxes", pred_boxes)
    # model = YOLOv1_resnet18().to(device)
    # pred_boxes, true_boxes = get_bboxes(train_loader,model)

