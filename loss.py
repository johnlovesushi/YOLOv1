import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from yolov1 import YOLOv1
from torchsummary import summary
import config
import random
from voc import VOCDataset
import utils
from darknet import DarkNet
from tqdm import tqdm

class Loss(nn.Module):
    
    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(Loss, self).__init__()
        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, p,a):
        """
        Args:
            p: (Tensor) bounding bboxes, sized [N,S,S,B*2 + C]
            a: (Tensor) bounding bboxes, sized [N,S,S,B*2 + C]
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        p_tl, p_br = self.bbox_to_coords(p)          # (batch, S, S, B, 2)
        a_tl, a_br = self.bbox_to_coords(a)
        coords_join_size = (-1, -1, -1, config.B, config.B, 2) 

        tl = torch.max(
            p_tl.unsqueeze(4).expand(coords_join_size),  # (batch,S,S,B,1,2) -> (batch,S,S,B,2,2)
            a_tl.unsqueeze(3).expand(coords_join_size)   # (batch, S,S,1,B,2) -> (batch, S,S,1,B,2)
        )
        br = torch.min(
            p_br.unsqueeze(4).expand(coords_join_size),  # (batch,S,S,B,1,2) -> (batch,S,S,B,2,2)
            a_br.unsqueeze(3).expand(coords_join_size)   # (batch, S,S,1,B,2) -> (batch, S,S,1,B,2)
        )

        intersection_sides = torch.clamp(br - tl, min=0.0)
        intersection = intersection_sides[..., 0] \
                    * intersection_sides[..., 1]       # (batch, S, S, B, B)


        p_area = self.bbox_attr(p, 2) * self.bbox_attr(p, 3)  # (batch,S,S,B)
        a_area = self.bbox_attr(a, 2) * self.bbox_attr(a, 3)  # (batch,S,S,B)

        p_area = p_area.unsqueeze(4).expand_as(intersection)     # (batch,S,S,B) -> (batch,S,S,B,1) ->  (batch,S,S,B,B) 
        a_area = a_area.unsqueeze(3).expand_as(intersection)     # (batch,S,S,B) -> (batch,S,S,1,B) ->  (batch,S,S,B,B) 

        union = p_area + a_area - intersection
        
        # catch division-by-zero
        zero_unions = (union == 0.0)
        union[zero_unions] = config.EPSILON
        intersection[zero_unions] = 0.0
        iou = intersection/ union
        
        return iou
    def mse_loss(self,a, b):
        flattened_a = torch.flatten(a, end_dim=-2)
        flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
        return F.mse_loss(
            flattened_a,
            flattened_b,
            reduction='sum'
        )

    def forward(self, p, a):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        S,B,C = self.S, self.B, self.C
        # loss includes 5 sub-losses: 
        # 1. re-construe to [N,S,S,B,5]
        iou = self.compute_iou(p,a) # (batch,S,S,B,B) 
        max_iou = torch.max(iou, dim = -1)[0]   # (batch,S,S,B) 

        bbox_mask = self.bbox_attr(a, 4) > 0.0  # (batch,S,S,B)
        p_template = self.bbox_attr(p, 4) > 0.0 # (batch,S,S,B)

        obj_i = bbox_mask[..., 0:1] # (batch, S,S,1)
        responsible = torch.zeros_like(p_template).scatter_(       # (batch, S, S, B)
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),                # (batch, S, S, B)
            value=1                         # 1 if bounding box is "responsible" for predicting the object
        )

        obj_ij = obj_i * responsible        # 1 if object exists AND bbox is responsible
        noobj_ij = ~obj_ij                  # Otherwise, confidence should be 0

        loss_x = self.mse_loss(obj_ij * self.bbox_attr(a,0), obj_ij * self.bbox_attr(p,0) )
        loss_y = self.mse_loss(obj_ij * self.bbox_attr(a,1), obj_ij * self.bbox_attr(p,1) )
        loss_w = self.mse_loss(obj_ij * torch.sign(self.bbox_attr(a,2)) * (self.bbox_attr(a,2)+config.EPSILON).sqrt(), 
                               obj_ij * torch.sign(self.bbox_attr(p,2)) * (self.bbox_attr(p,2)+config.EPSILON).sqrt())
        loss_h = self.mse_loss(obj_ij * torch.sign(self.bbox_attr(a,3)) * (self.bbox_attr(a,3)+config.EPSILON).sqrt(), 
                               obj_ij * torch.sign(self.bbox_attr(p,3)) * (self.bbox_attr(p,3)+config.EPSILON).sqrt())

        #

        loss_conf_obj = self.mse_loss(obj_ij * obj_ij * torch.ones_like(max_iou),obj_ij * self.bbox_attr(p,4))
        loss_conf_no_obj = self.mse_loss(noobj_ij * obj_ij * torch.ones_like(max_iou),noobj_ij * self.bbox_attr(p,4))

        classification_loss = self.mse_loss(obj_i * p[...,5 * config.B: ] , obj_i * a[...,5 * config.B: ])


        total_loss = (
            self.lambda_coord * (loss_x + loss_y + loss_w + loss_h) +
            loss_conf_obj +
            self.lambda_noobj * loss_conf_no_obj +
            classification_loss
        )
        return total_loss

    def bbox_to_coords(self,t):
        """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

        width = self.bbox_attr(t, 2)
        x = self.bbox_attr(t, 0)
        x1 = x - width / 2.0
        x2 = x + width / 2.0

        height = self.bbox_attr(t, 3)
        y = self.bbox_attr(t, 1)
        y1 = y - height / 2.0
        y2 = y + height / 2.0

        return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)

    def bbox_attr(self,data, i):
        """Returns the Ith attribute of each bounding box in data."""

        attr_start = config.C + i
        return data[..., attr_start::5]

def test_compute_iou():
    # Example test
    test_loss = Loss()
    bbox1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])  # Shape [2, 4]
    bbox2 = torch.tensor([[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0]])  # Shape [2, 4]
    iou_matrix = test_loss.compute_iou(bbox1, bbox2)
    print(iou_matrix)
    expected_res = torch.tensor([[0.2500, 0.0000],[0.1429, 0.0000]])
    print(f"Example 1 IoU Matrix test:\{expected_res == iou_matrix}")

    # expected: tensor([[0.2500, 0.0000],[0.1429, 0.0000]])
    # Example with some overlap and some no overlap
    bbox1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0]])  # Shape [2, 4]
    bbox2 = torch.tensor([[1.0, 1.0, 3.0, 3.0], [3.0, 3.0, 5.0, 5.0]])  # Shape [2, 4]
    iou_matrix = test_loss.compute_iou(bbox1, bbox2)
    expected_res = torch.tensor([[0.1429, 0.0000],[0.1429, 0.1429]])
    print(f"Example 2 IoU Matrix test:\{expected_res == iou_matrix}")

    # expected IoU Matrix:


    # bbox1 contains 3 bounding boxes
    bbox1 = torch.tensor([[0.0, 0.0, 1.0, 1.0],  # Box 1
                        [1.0, 1.0, 3.0, 3.0],  # Box 2
                        [2.0, 2.0, 4.0, 4.0]]) # Box 3

    # bbox2 contains 4 bounding boxes
    bbox2 = torch.tensor([[0.5, 0.5, 1.5, 1.5],  # Box 1
                        [2.0, 2.0, 3.0, 3.0],  # Box 2
                        [3.0, 3.0, 5.0, 5.0],  # Box 3
                        [0.0, 0.0, 2.0, 2.0]]) # Box 4

    iou_matrix = test_loss.compute_iou(bbox1, bbox2)
    print(iou_matrix)
    expected_res = torch.tensor([[0.1429, 0.0000, 0.0000, 0.2500],
                                [0.0526, 0.2500, 0.0000, 0.1429],
                                [0.0000, 0.2500, 0.1429, 0.0000]])
    print(f"Example 3 IoU Matrix test:\{expected_res == iou_matrix}")

    # IoU Matrix:
    # tensor([[0.1429, 0.0000, 0.0000, 0.2500],
    #         [0.0526, 0.2500, 0.0000, 0.1429],
    #         [0.0000, 0.2500, 0.1429, 0.0000]])

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)  # Stack images
    targets = torch.stack([item[1] for item in batch], dim=0)  # Stack targets
    return images, targets

def test_loss_function():
    from torch.utils.data import Subset
    random.seed(10)
    EPOCH = 5
    init_lr = 0.001
    momentum = 0.9
    weight_decay = 5.0e-4
    dataset = VOCDataset(is_train = True, normalize = True)
    subset_size = 100  # taking 10 samples
    dataset_size = len(dataset)
    # Randomly select indices for the subset
    subset_indices = random.sample(range(dataset_size), subset_size)
    subset = Subset(dataset, subset_indices)

    # for data,target,label,_ in subset:
    #     obj_classes = utils.load_class_array()
    #     file = "./output_images"
    #     utils.plot_boxes(data, target,label, obj_classes, max_overlap=float('inf'), file = file)
    darknet = DarkNet(conv_only=True, batch_norm=True, init_weight=True)
    yolo = YOLOv1(darknet.features).to('cpu')
    dataloader = DataLoader(subset, batch_size=4, shuffle=True,collate_fn=custom_collate_fn)
    criterion = Loss(feature_size=yolo.num_grids)
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    # for image in subset:
    #     print(image.shape)
    # for images,targets in dataloader:
    #     print(images.shape)
    for epoch in range(EPOCH):
        yolo.train()  # Set the model to training mode for each epoch
        epoch_loss = 0.0  # Initialize the loss for the epoch
        total_train = 0
        correct_train = 0
        num_batches = len(dataloader)  # Get the number of batches

        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{EPOCH}", unit="batch") as pbar:
            for batch_idx, (images,targets) in enumerate(dataloader):

                images, targets = images.to('cpu'), targets.to('cpu')
                # print(f"Input type: {images.dtype}, shape: {images.shape}")
                # print(f"Input type: {targets.dtype}, shape: {targets.shape}")
                predicted = yolo(images)  # Forward pass through the model

                # print(f"Outputs shape: {outputs.shape}")  # Check the shape of outputs
                # print(f"Targets shape: {targets.shape}")  # Check the shape of targets

                loss = criterion(predicted, targets)  # Compute the loss
                optimizer.zero_grad()  # Clear the gradients
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                epoch_loss += loss.item()  # Accumulate the loss

if __name__ == '__main__':
   #test_compute_iou()
   test_loss_function()