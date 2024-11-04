import torch
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset,  DataLoader
from models.yolov1net_resnet18 import YOLOv1_resnet18
from mean_average_precision import MetricBuilder
import albumentations as albu
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VOCDataset(Dataset):

    def __init__(self, is_train = True, normalize = False):
        self.is_train = is_train
        self.normalize = normalize
        self.dataset = VOCDetection(
            root = config.DATA_PATH,
            year = '2007',
            image_set = ("train" if self.is_train else "val"),
            download = True,
            transform = T.Compose([
                T.Resize(config.IMAGE_SIZE),
                T.ToTensor()
            ])
        )

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
        data, label = self.dataset[index]
        original_data = data.clone()
        file_name = label['annotation']['filename']
        #original_data = data

        difficult =torch.as_tensor([int(detection['difficult']) for detection in label['annotation']['object']])
        #print(difficult)
        labels_tensor = torch.as_tensor([self.classes[detection['name']] for detection in label['annotation']['object']])
        bboxes_tensor = torch.as_tensor([[float(detection['bndbox']['xmin']), 
                                          float(detection['bndbox']['ymin']),
                                          float(detection['bndbox']['xmax']), 
                                          float(detection['bndbox']['ymax'])] for detection in label['annotation']['object']])
        #print(bboxes_tensor)
        w,h = float(label['annotation']['size']['width']), float(label['annotation']['size']['height'])
        bboxes_tensor /= torch.tensor([w,h,w,h], dtype = torch.float32).expand_as(bboxes_tensor)
        #print('bboxes_tensor',bboxes_tensor)
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()
        if self.is_train:
            # scale, flip,blur,brightness,hue,saturation，crop
            # scale
            data = TF.affine(data,angle=0.0,translate=(x_shift, y_shift),scale=scale,shear=0.0)
            # blur
            data = TF.gaussian_blur(data,kernel_size = random.choice([1, 3, 5]) )
            # brightness
            data = TF.adjust_brightness(data,brightness_factor = random.uniform(0.5, 1.5))
            # hue
            data = TF.adjust_hue(data,hue_factor = random.random()*0.2 - 0.1)
            # saturation
            data = TF.adjust_saturation(data,saturation_factor = 0.2 * random.random() + 0.9)
        
        if self.normalize: # Keep it so far
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        yolo_targets,adjusted_bboxes_tensor = self._encode(data, label,scale,x_shift, y_shift)
        adjusted_bboxes_tensor /= torch.tensor([config.IMAGE_SIZE[0],config.IMAGE_SIZE[1],config.IMAGE_SIZE[0],config.IMAGE_SIZE[1]], dtype = torch.float32).expand_as(bboxes_tensor)
        #print('yolo_targets',yolo_targets)
        targets = {
                    'bboxes': bboxes_tensor,
                    'adjusted_bboxes':adjusted_bboxes_tensor,
                    'labels': labels_tensor,
                    'yolo_targets': yolo_targets,
                    'difficult': difficult,
                    'original_data': original_data,
                    
                }
        return data, targets, file_name #label
    
    def _encode(self, data, label,scale,x_shift, y_shift):

        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        N = 5 * config.B + config.C     # 5 numbers per bbox, then one-hot encoding of label

        grid_size_x = data.shape[2]/config.S
        grid_size_y = data.shape[1]/config.S
        target = torch.zeros((config.S, config.S, N))
        adjusted_bboxes = []
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min,x_max,y_min,y_max = coords
            
            if self.is_train:
                            # Augment labels
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift
            # Keep the record of the adjusted_bboes coord
            adjusted_bboxes.append([x_min,y_min,x_max,y_max])
        
        
        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)            

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    target[row, col, :config.C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],     # X coord relative to grid square
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],     # Y coord relative to grid square
                            (x_max - x_min) / config.IMAGE_SIZE[0],                 # Width
                            (y_max - y_min) / config.IMAGE_SIZE[1],                 # Height
                            1.0                                                     # Confidence
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 5 * bbox_index + config.C
                        target[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1
        adjusted_bboxes_tensor = torch.as_tensor(adjusted_bboxes)
        return target,adjusted_bboxes_tensor

    def __len__(self):
        return len(self.dataset)
def collate_function(data):
    return list(zip(*data))


def test():
    from torch.utils.data import Subset
    import pprint
    torch.set_printoptions(profile="full")
    random.seed(10)

    dataset = VOCDataset(is_train = True, normalize = True)
    subset_size = 20  # taking 10 samples
    dataset_size = len(dataset)
    # Randomly select indices for the subset
    subset_indices = random.sample(range(dataset_size), subset_size)
    subset = Subset(dataset, subset_indices)
    # batch_size can only be 1 when its evaluate mAP
    dataloader =  DataLoader(subset, shuffle=True, batch_size = 1, collate_fn=collate_function)
    yolo = YOLOv1_resnet18().to(device)
    for idx, (data,targets, filename) in enumerate(dataloader):
        #print(data)
        yolo_targets = torch.cat([
            target['yolo_targets'].unsqueeze(0).float().to(device)
            for target in targets], dim=0)
        im = torch.cat([im.unsqueeze(0).float().to(device) for im in data], dim=0)
        #target_bboxes = targets['bboxes'].to(device)[0]
        # bboxes = torch.cat([
        #     target['bboxes'].unsqueeze(0).float().to(device)
        #     for target in targets], dim=0)
        labels = torch.cat([
            target['labels'].unsqueeze(0).float().to(device)
            for target in targets], dim=0)
        #difficult = targets['difficult']
        difficult = torch.cat([
            target['difficult'].unsqueeze(0).float().to(device)
            for target in targets], dim=0)
        # bboxes = targets['bboxes'].float().to(device)[0]
        # labels = targets['labels'].long().to(device)[0]
        # difficult = targets['difficult'].long().to(device)[0]
        pred = yolo(im)
        bboxes = targets[0]['bboxes'].float().to(device)
        labels = targets[0]['labels'].float().to(device)
        difficult = targets[0]['difficult'].float().to(device)
        adjusted_bboxes = targets[0]['adjusted_bboxes'].float().to(device)
        print(f"Batch {idx + 1}:")
        print("bboxes", bboxes)
        print('adjusted_bboxes', adjusted_bboxes)
        print("labels", labels)
        print("difficult", difficult)
        print("filename", filename)

        # print("Data:", data)  # Display shape of data tensor
        # print("Targets:", targets)  # Display targets dictionary
        # print("label:", bboxes)  # Display filenames
        # print("bboxes",labels)
        # Print each component in the targets dictionary to check values
        pred = utils.mean_average_precision(pred,conf_threshold = 0.3,iou_threshold = 0.3)
        #print("pred",pred)
        tgs = utils.convert_target_to_certain_format(adjusted_bboxes, labels,difficult)
        #print("tgs", tgs)

        #print(MetricBuilder.get_metrics_list())
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=config.C)
        metric_fn.add(pred, tgs)
        # compute PASCAL VOC metric
        print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
        if idx == 10:
            break
        
        # # Now you can work with each batch element
        # print("Data shape:", data.shape)  # Transformed images after augmentations
        # print("Target:", target.shape)          # Encoded targets for model
        # print("Label:", label)            # Original label information
        # print("Original Data shape:", original_data.shape)  # Untransformed images
         # Remove this break in actual training loop
    # for data,target,label,_ in subset:
    #     obj_classes = utils.load_class_array()
    #     file = "./output_images"
    #     utils.plot_boxes(data, target,label, obj_classes, max_overlap=float('inf'), file = file)        
    #     print(target)
if __name__ == '__main__':
    test()