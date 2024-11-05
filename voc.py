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
        class_names = {}                # Track what class each grid cell has been assigned to
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
        adjusted_bboxes_tensor = torch.as_tensor(adjusted_bboxes, dtype = torch.float32)
        return target,adjusted_bboxes_tensor

    def __len__(self):
        return len(self.dataset)