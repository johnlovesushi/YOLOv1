import torch
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset,  DataLoader, Subset
from models.yolov1net_resnet18 import YOLOv1_resnet18
from scale_image import scale_image,scale_translate_bounding_box
from utils import convert_bboxes_entire_img_ratios,convert_bboxes_to_list,get_bboxes,mean_average_precision
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VOCDataset(Dataset):

    def __init__(self, is_train = True, transform = None, transform_scale_translate = True):
        self.is_train = is_train
        self.dataset = VOCDetection(
            root = config.DATA_PATH,
            year = '2007',
            image_set = ("train" if self.is_train else "val"),
            download = True
            #transform= T.ToTensor()
        )
        self.transform = transform
        self.transform_scale_translate = transform_scale_translate
        self.classes = utils.load_class_dict()

        
        # Generate class index if needed
        index = 0
        if len(self.classes) == 0:
            for i, data_pair in enumerate(tqdm(self.dataset, desc=f'Generating class dict')):
                data, label = data_pair
                for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
                    name, coords = bbox_pair
                    if name not in self.classes:
                        self.classes[name] = index
                        index += 1
            utils.save_class_dict(self.classes)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        file_name = label['annotation']['filename']
        difficult =torch.as_tensor([int(detection['difficult']) for detection in label['annotation']['object']])

        labels_tensor = torch.as_tensor([self.classes[detection['name']] for detection in label['annotation']['object']])
        bboxes_xyxy_tensor = torch.as_tensor([[float(detection['bndbox']['xmin']), 
                                          float(detection['bndbox']['ymin']),
                                          float(detection['bndbox']['xmax']), 
                                          float(detection['bndbox']['ymax'])] for detection in label['annotation']['object']], dtype = float)
        
        w,h = float(label['annotation']['size']['width']), float(label['annotation']['size']['height'])
        bboxes_xywh_tensor = utils.x1y1x2y2_convert_to_xywh(bboxes_xyxy_tensor)
        scaled_bboxes_xywh_tensor = bboxes_xywh_tensor/torch.tensor([w,h,w,h], dtype = torch.float32).expand_as(bboxes_xyxy_tensor)
        scaled_bboxes_xyxy_tensor = bboxes_xyxy_tensor/torch.tensor([w,h,w,h], dtype = torch.float32).expand_as(bboxes_xyxy_tensor)      
        if self.transform_scale_translate == True:
            img_scale_trans, transform_vals = scale_image(image)
            scaled_bboxes_xywh_tensor = scale_translate_bounding_box(scaled_bboxes_xywh_tensor, transform_vals)
        
        if self.transform:
            data, scaled_bboxes_xywh_tensor = self.transform(img_scale_trans, scaled_bboxes_xywh_tensor)


        targets = self._encode(scaled_bboxes_xywh_tensor, labels_tensor)
        return data, targets#file_name #label
    
    def _encode(self, scaled_bboxes, label):

        class_names = {}                # Track what class each grid cell has been assigned to
        N = config.C + 5 * config.B      # 5 numbers per bbox, then one-hot encoding of label

        target = torch.zeros((config.S, config.S, N))

        num_of_boxes = scaled_bboxes.shape[0]
        for i in range(num_of_boxes):
            class_label = label[i]
            x, y, h, w = scaled_bboxes[i,...]
            coord_x, coord_y = int(config.S * x), int(config.S * y)
            cell_x,cell_y = config.S * x - coord_x, config.S * y - coord_y

            cell_w, cell_h = config.S * w ,config.S * h

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if target[coord_y,coord_x,  20] == 0:
                # set this
                target[coord_y,coord_x,  20] = 1

                target[coord_y,coord_x,  21:25] = torch.tensor([cell_x,cell_y,cell_w, cell_h])
                target[coord_y,coord_x, class_label] = 1

        return target

    def __len__(self):
        return len(self.dataset)

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
    

def test():
    torch.set_printoptions(threshold=float("inf"))
    nworkers = 2


    train_dataset = VOCDataset(is_train = True,transform = train_transform, transform_scale_translate = True)

    subset_size = 10  
    dataset_size = len(train_dataset)
    subset_indices = range(1,subset_size)
    subset = Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(dataset = subset, batch_size = 1, 
                              num_workers = 0, shuffle = False)
    model = YOLOv1_resnet18().to(device)
    pred_boxes, true_boxes = get_bboxes(train_loader,model)

    mAP = mean_average_precision(pred_boxes, true_boxes)
    print("mAP",mAP)
    for batch_idx, (data,targets) in enumerate(train_loader):
        data= data.to(device)
        targets = targets.to(device)
        preds = model(data)
        # print('target',convert_bboxes_entire_img_ratios(targets))
        # print('pred',convert_bboxes_entire_img_ratios(preds))
        # print(convert_bboxes_to_list(targets))
        # print(convert_bboxes_to_list(preds))
        #print('file_name', file_name)
        #print('targets',targets['yolo_targets'])
        # print('imgs',imgs)
        # print('tgs',tgs)
        if batch_idx == 0:
            break

if __name__ == '__main__':
    test()