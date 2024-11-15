

import torch
import torch.nn as nn
#from utils.yolov1_utils import intersec_over_union as IUO

import torch 
from collections import Counter
device = "cuda" if torch.cuda.is_available() else "cpu"

def IUO(bboxes_preds, bboxes_targets, boxformat = "midpoints"):    
    """
    Calculates intersection of unions (IoU).
    Input: Boundbing box predictions (tensor) x1, x2, y1, y2 of shape (N , 4)
            with N denoting the number of bounding boxes.
            Bounding box target/ground truth (tensor) x1, x2, y1, y2 of shape (N, 4).
            box format whether midpoint location or corner location of bounding boxes
            are used.
    Output: Intersection over union (tensor).
    """
    
    if boxformat == "midpoints":
        box1_x1 = bboxes_preds[...,0:1] - bboxes_preds[...,2:3] / 2
        box1_y1 = bboxes_preds[...,1:2] - bboxes_preds[...,3:4] / 2
        box1_x2 = bboxes_preds[...,0:1] + bboxes_preds[...,2:3] / 2
        box1_y2 = bboxes_preds[...,1:2] + bboxes_preds[...,3:4] / 2
    
        box2_x1 = bboxes_targets[...,0:1] - bboxes_targets[...,2:3] / 2
        box2_y1 = bboxes_targets[...,1:2] - bboxes_targets[...,3:4] / 2
        box2_x2 = bboxes_targets[...,0:1] +  bboxes_targets[...,2:3] / 2
        box2_y2 = bboxes_targets[...,1:2] +  bboxes_targets[...,3:4] / 2
        
    if boxformat == "corners":
        box1_x1 = bboxes_preds[...,0:1]
        box1_y1 = bboxes_preds[...,1:2]
        box1_x2 = bboxes_preds[...,2:3]
        box1_y2 = bboxes_preds[...,3:4]
    
        box2_x1 = bboxes_targets[...,0:1]
        box2_y1 = bboxes_targets[...,1:2]
        box2_x2 = bboxes_targets[...,2:3]
        box2_y2 = bboxes_targets[...,3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # clip intersection at zero to ensure it is never negative and equal to zero
    # if no intersection exists
    intersec = torch.clip((x2 - x1), min = 0) * torch.clip((y2 - y1), min = 0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersec + 1e-6
    iou = intersec / union
    return iou

class YoloV1Loss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_no_obj = 0.5
        self.lambda_obj = 5



    def forward(self, preds, target):
        mse_loss = nn.MSELoss(reduction="sum")
        # reshape predictions to S by S by 30
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # extract 4 bounding box values for bounding box 1 and box 2
        iou_bbox1 = IUO(preds[...,21:25], target[...,21:25])
        iou_bbox2 = IUO(preds[...,26:30], target[...,21:25])
        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim = 0)
        _ , bestbox = torch.max(ious, dim = 0)
        # Determine if an object is in cell i using identity
        identity_obj_i = target[...,20].unsqueeze(3) 

        # 1. Bouding Box Loss Component 
        boxpreds = identity_obj_i * (
            (
                bestbox * preds[...,26:30] 
                + (1 - bestbox) * preds[...,21:25]
            )
        )
        
        boxtargets = identity_obj_i * target[...,21:25]

        boxpreds[...,2:4] = torch.sign(boxpreds[...,2:4]) * torch.sqrt(
            torch.abs(boxpreds[...,2:4] + 1e-6)
        )    
        boxtargets[...,2:4] = torch.sqrt(boxtargets[...,2:4])
        
        # N, S, S, 4 -> N*N*S,4

        boxloss = mse_loss(torch.flatten(boxpreds, end_dim = -2),
                           torch.flatten(boxtargets, end_dim = -2)
        )
        
        # 2. Object Loss Component
        # has shape N by S by S

        predbox = (
            bestbox * preds[...,25:26] + (1 - bestbox) * preds[...,20:21]
            )
        
        
        objloss = mse_loss(
            torch.flatten(identity_obj_i * predbox),
            torch.flatten(identity_obj_i * target[...,20:21])
        )
        
        
        # 3. No Object Loss Component 
        no_objloss = mse_loss(
            torch.flatten((1 - identity_obj_i) * preds[...,20:21], start_dim = 1),
            torch.flatten((1 - identity_obj_i) * target[...,20:21], start_dim = 1)
            )
        
        no_objloss += mse_loss(
            torch.flatten((1 - identity_obj_i) * preds[...,25:26], start_dim = 1),
            torch.flatten((1 - identity_obj_i) * target[...,20:21], start_dim = 1)
            )
        
        # 4. Class Loss Component 
        classloss = mse_loss(
            torch.flatten(identity_obj_i * preds[...,:20], end_dim = -2),
            torch.flatten(identity_obj_i * target[...,:20], end_dim = -2)
            )
        
        # 5. Combine Loss Components 
        loss = (
            self.lambda_obj * boxloss
            + objloss
            + self.lambda_no_obj * no_objloss
            + classloss)

        return loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from models.yolov1net_resnet18 import YOLOv1_resnet18
# from torchsummary import summary
# import config
# import random
# from voc import VOCDataset
# from utils import *
# from tqdm import tqdm
# from utils import mean_average_precision as mAP
# from utils import convert_yolo_pred_x1y1x2y2


# def collate_function(data):
#     return list(zip(*data))

# class Loss(nn.Module):
    
#     def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
#         super(Loss, self).__init__()
#         self.S = feature_size
#         self.B = num_bboxes
#         self.C = num_classes
#         self.lambda_coord = lambda_coord
#         self.lambda_noobj = lambda_noobj

#     def mse_loss(self,a, b):
#         flattened_a = torch.flatten(a, end_dim=-2)
#         flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
#         return F.mse_loss(flattened_a,flattened_b,reduction='sum')

#     def forward(self, p, a):
#         """ Compute loss for YOLO training.
#         Args:
#             pred_tensor: (Tensor) predictions, sized [n_batch, S, S, C+Bx5], 5=len([conf,x, y, w, h]).
#             target_tensor: (Tensor) targets, sized [n_batch, S, S, C+Bx5].
#         Returns:
#             (Tensor): loss, sized [1, ].
#         """
#         S,B,C = self.S, self.B, self.C
#         # loss includes 5 sub-losses: 
#         # 1. re-construe to [N,S,S,B,5]
#         iou = compute_iou(p,a) # (batch,S,S,B,B) 
#         max_iou = torch.max(iou, dim = -1)[0]   # (batch,S,S,B) 

#         bbox_mask = bbox_attr(a, 0) > 0.0  # conf (batch,S,S,B)
#         p_template = bbox_attr(p, 0) > 0.0 # conf (batch,S,S,B)

#         obj_i = bbox_mask[..., 0:1] # (batch, S,S,1)
#         responsible = torch.zeros_like(p_template).scatter_(       # (batch, S, S, B)
#             -1,
#             torch.argmax(max_iou, dim=-1, keepdim=True),                # (batch, S, S, B)
#             value=1                         # 1 if bounding box is "responsible" for predicting the object
#         )

#         obj_ij = obj_i * responsible        # 1 if object exists AND bbox is responsible
#         noobj_ij = ~obj_ij                  # Otherwise, confidence should be 0

#         loss_x = self.mse_loss(obj_ij * bbox_attr(a,1), obj_ij * bbox_attr(p,1) )
#         loss_y = self.mse_loss(obj_ij * bbox_attr(a,2), obj_ij * bbox_attr(p,2) )
#         loss_w = self.mse_loss(obj_ij * torch.sign(bbox_attr(a,3)) * (bbox_attr(a,3)+config.EPSILON).sqrt(), 
#                                obj_ij * torch.sign(bbox_attr(p,3)) * (bbox_attr(p,3)+config.EPSILON).sqrt())
#         loss_h = self.mse_loss(obj_ij * torch.sign(bbox_attr(a,4)) * (bbox_attr(a,4)+config.EPSILON).sqrt(), 
#                                obj_ij * torch.sign(bbox_attr(p,4)) * (bbox_attr(p,4)+config.EPSILON).sqrt())

#         #

#         loss_conf_obj = self.mse_loss(obj_ij * torch.ones_like(max_iou),obj_ij * bbox_attr(p,0))
#         loss_conf_no_obj = self.mse_loss(noobj_ij  * torch.ones_like(max_iou),noobj_ij * bbox_attr(p,0))

#         classification_loss = self.mse_loss(obj_i * p[...,: config.C] , obj_i * a[...,: config.C])


#         total_loss = (
#             self.lambda_coord * (loss_x + loss_y + loss_w + loss_h) +
#             loss_conf_obj +
#             self.lambda_noobj * loss_conf_no_obj +
#             classification_loss
#         )
#         return total_loss

# def test_compute_iou():
#     # Example test
#     test_loss = Loss()
#     bbox1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])  # Shape [2, 4]
#     bbox2 = torch.tensor([[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0]])  # Shape [2, 4]
#     iou_matrix = test_loss.compute_iou(bbox1, bbox2)
#     print(iou_matrix)
#     expected_res = torch.tensor([[0.2500, 0.0000],[0.1429, 0.0000]])
#     print(f"Example 1 IoU Matrix test:\{expected_res == iou_matrix}")

#     # expected: tensor([[0.2500, 0.0000],[0.1429, 0.0000]])
#     # Example with some overlap and some no overlap
#     bbox1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0]])  # Shape [2, 4]
#     bbox2 = torch.tensor([[1.0, 1.0, 3.0, 3.0], [3.0, 3.0, 5.0, 5.0]])  # Shape [2, 4]
#     iou_matrix = test_loss.compute_iou(bbox1, bbox2)
#     expected_res = torch.tensor([[0.1429, 0.0000],[0.1429, 0.1429]])
#     print(f"Example 2 IoU Matrix test:\{expected_res == iou_matrix}")

#     # expected IoU Matrix:


#     # bbox1 contains 3 bounding boxes
#     bbox1 = torch.tensor([[0.0, 0.0, 1.0, 1.0],  # Box 1
#                         [1.0, 1.0, 3.0, 3.0],  # Box 2
#                         [2.0, 2.0, 4.0, 4.0]]) # Box 3

#     # bbox2 contains 4 bounding boxes
#     bbox2 = torch.tensor([[0.5, 0.5, 1.5, 1.5],  # Box 1
#                         [2.0, 2.0, 3.0, 3.0],  # Box 2
#                         [3.0, 3.0, 5.0, 5.0],  # Box 3
#                         [0.0, 0.0, 2.0, 2.0]]) # Box 4

#     iou_matrix = test_loss.compute_iou(bbox1, bbox2)
#     print(iou_matrix)
#     expected_res = torch.tensor([[0.1429, 0.0000, 0.0000, 0.2500],
#                                 [0.0526, 0.2500, 0.0000, 0.1429],
#                                 [0.0000, 0.2500, 0.1429, 0.0000]])
#     print(f"Example 3 IoU Matrix test:\{expected_res == iou_matrix}")

#     # IoU Matrix:
#     # tensor([[0.1429, 0.0000, 0.0000, 0.2500],
#     #         [0.0526, 0.2500, 0.0000, 0.1429],
#     #         [0.0000, 0.2500, 0.1429, 0.0000]])

# def custom_collate_fn(batch):
#     images = torch.stack([item[0] for item in batch], dim=0)  # Stack images
#     targets = torch.stack([item[1] for item in batch], dim=0)  # Stack targets
#     return images, targets

# def test_loss_function():
#     from torch.utils.data import Subset
#     random.seed(10)
#     EPOCH = 5
#     init_lr = 0.001
#     momentum = 0.9
#     weight_decay = 5.0e-4
#     dataset = VOCDataset(is_train = True, normalize = True)
#     subset_size = 20  # taking 10 samples
#     dataset_size = len(dataset)
#     # Randomly select indices for the subset
#     subset_indices = random.sample(range(dataset_size), subset_size)
#     subset = Subset(dataset, subset_indices)

#     # for data,target,label,_ in subset:
#     #     obj_classes = utils.load_class_array()
#     #     file = "./output_images"
#     #     utils.plot_boxes(data, target,label, obj_classes, max_overlap=float('inf'), file = file)
#     #darknet = DarkNet(conv_only=True, batch_norm=True, init_weight=True)
#     yolo = YOLOv1_resnet18().to('cpu')
#     dataloader = DataLoader(subset, batch_size=4, shuffle=True) #,collate_fn=custom_collate_fn
#     # for batch in dataloader:
#     #     data, target, original_data = batch
#     #     # Now you can work with each batch element
#     #     print("Data shape:", data.shape)  # Transformed images after augmentations
#     #     print("Target:", target.shape)          # Encoded targets for model
#     #     print("Original Data shape:", original_data.shape)  # Untransformed images
#     #     break  # Remove this break in actual training loop
#     # criterion = Loss(feature_size=yolo.num_grids)
#     optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
#     torch.set_printoptions(profile="full")

#     for data,target,_ in subset:
#         obj_classes = load_class_array()
#         file = "./output_images"
#         pred = yolo(data.unsqueeze(0))
#         # plot_boxes(data, target,label, obj_classes, max_overlap=float('inf'), file = file)        
#         # print(target)
#         # true_bboxes = convert_yolo_pred_x1y1x2y2(target.unsqueeze(0), S = 7,B=2, C=20)       # [batch_size, S*S]
#         # print(true_bboxes)
#         # bboxes = convert_yolo_pred_x1y1x2y2(predicted, S = 7,B=2, C=20)
#         # print('true_bboxes',true_bboxes)
#         # print(bboxes)
#         # print(len(bboxes), len(bboxes[0]))
#         # #for training data
#         mAP(pred,target,conf_threshold = 0.3,iou_threshold = 0.3)

#     pred_bbox, target_bbox = get_bboxes(dataloader, yolo, iou_threshold = 0.5, 
#                                           threshold = 0.4)
    
#     for epoch in range(EPOCH):
#         train_mAP_val = mAP(pred_bbox, target_bbox, iou_threshold = 0.5, boxformat="midpoints")
#         yolo.train()  # Set the model to training mode for each epoch
#         epoch_loss = 0.0  # Initialize the loss for the epoch
#         total_train = 0
#         correct_train = 0
#         num_batches = len(dataloader)  # Get the number of batches


#         with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{EPOCH}", unit="batch") as pbar:
#             for batch_idx, batch in enumerate(dataloader):
#                 images, targets, _,_ = batch
#                 images, targets = images.to('cpu'), targets.to('cpu')
#                 # print(f"Input type: {images.dtype}, shape: {images.shape}")
#                 # print(f"Input type: {targets.dtype}, shape: {targets.shape}")
#                 predicted = yolo(images)  # Forward pass through the model
#                 print(images)
#                 true_bboxes = cellboxes_to_boxes(targets)       # [batch_size, S*S]
#                 bboxes = cellboxes_to_boxes(predicted)
#                 print('true_bboxes',true_bboxes)
#                 print(len(true_bboxes), len(true_bboxes[0]), )
#                 print(bboxes)
#                 print(len(bboxes), len(bboxes[0]))
#                 # print(f"Outputs shape: {outputs.shape}")  # Check the shape of outputs
#                 # print(f"Targets shape: {targets.shape}")  # Check the shape of targets

#                 loss = criterion(predicted, targets)  # Compute the loss
#                 optimizer.zero_grad()  # Clear the gradients
#                 loss.backward()  # Backpropagation
#                 optimizer.step()  # Update weights
#                 convert_cellboxes(predicted)
#                 epoch_loss += loss.item()  # Accumulate the loss
#                 pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
#                 pbar.update(1)  # Update the progress bar by 1 batch

#                 # Calculate train loss
#         avg_train_loss = epoch_loss / num_batches

#         # Log training metrics to TensorBoard
#         #writer.add_scalar('Train/Loss', avg_train_loss, epoch)

#       # After each epoch, print summary
#         print(f"Epoch [{epoch + 1}/{EPOCH}] completed. Average Loss: {epoch_loss / num_batches:.4f}, mAP: {train_mAP_val}\n")



# if __name__ == '__main__':
#    #test_compute_iou()
#    test_loss_function()