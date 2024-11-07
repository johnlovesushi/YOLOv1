import torch
import json
import os
import config
# import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
# from matplotlib import pyplot as plt
# from collections import Counter
import numpy as np
from mean_average_precision import MetricBuilder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def compute_iou( p,a):
    """
    Args:
        p: (Tensor) bounding bboxes, sized [N,S,S,B*2 + C]
        a: (Tensor) bounding bboxes, sized [N,S,S,B*2 + C]
    Returns:
        (Tensor) IoU, sized [N, M].
    """
    p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)
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


    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)  # (batch,S,S,B)
    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)  # (batch,S,S,B)

    p_area = p_area.unsqueeze(4).expand_as(intersection)     # (batch,S,S,B) -> (batch,S,S,B,1) ->  (batch,S,S,B,B) 
    a_area = a_area.unsqueeze(3).expand_as(intersection)     # (batch,S,S,B) -> (batch,S,S,1,B) ->  (batch,S,S,B,B) 

    union = p_area + a_area - intersection
    
    # catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = config.EPSILON
    intersection[zero_unions] = 0.0
    iou = intersection/ union
    
    return iou

def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)

def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::5]
    
def scheduler_lambda(epoch):
    if epoch < config.WARMUP_EPOCHS + 75:
        return 1
    elif epoch < config.WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj):
    folder = os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


def get_dimensions(label):
    size = label['annotation']['size']
    return int(size['width']), int(size['height'])


def get_bounding_boxes(label):
    width, height = get_dimensions(label)
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label['annotation']['object']
    for obj in objects:
        box = obj['bndbox']
        coords = (
            int(int(box['xmin']) * x_scale),
            int(int(box['xmax']) * x_scale),
            int(int(box['ymin']) * y_scale),
            int(int(box['ymax']) * y_scale)
        )
        name = obj['name']
        boxes.append((name, coords))
    return boxes

def scale_bbox_coord(coord, center, scale):
    return ((coord - center) * scale) + center


def get_overlap(a, b):
    """Returns proportion overlap between two boxes in the form (tl, width, height, confidence, class)."""

    a_tl, a_width, a_height, _, _ = a
    b_tl, b_width, b_height, _, _ = b

    i_tl = (
        max(a_tl[0], b_tl[0]),
        max(a_tl[1], b_tl[1])
    )
    i_br = (
        min(a_tl[0] + a_width, b_tl[0] + b_width),
        min(a_tl[1] + a_height, b_tl[1] + b_height),
    )

    intersection = max(0, i_br[0] - i_tl[0]) \
                   * max(0, i_br[1] - i_tl[1])

    a_area = a_width * a_height
    b_area = b_width * b_height

    a_intersection = b_intersection = intersection
    if a_area == 0:
        a_intersection = 0
        a_area = config.EPSILON
    if b_area == 0:
        b_intersection = 0
        b_area = config.EPSILON

    return torch.max(
        a_intersection / a_area,
        b_intersection / b_area
    ).item()


def plot_boxes(data, targets, labels,classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None, output_folder = "./output_images"):
    """Plots bounding boxes on the given image."""

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S
    m = targets.size(dim=0)
    n = targets.size(dim=1)
    image_name = labels['annotation']['filename']
    bboxes = []
    for i in range(m):
        for j in range(n):
            for k in range((targets.size(dim=2) - config.C) // 5):
                bbox_start = 5 * k + config.C
                bbox_end = 5 * (k + 1) + config.C
                bbox = targets[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(targets[i, j, :config.C]).item()
                confidence = targets[i, j, class_index].item() * bbox[4].item()          # pr(c) * IOU
                if confidence > min_confidence:
                    width = bbox[2] * config.IMAGE_SIZE[0]
                    height = bbox[3] * config.IMAGE_SIZE[1]
                    tl = (
                        bbox[0] * config.IMAGE_SIZE[0] + j * grid_size_x - width / 2,
                        bbox[1] * config.IMAGE_SIZE[1] + i * grid_size_y - height / 2
                    )
                    bboxes.append([tl, width, height, confidence, class_index])

    # Sort by highest to lowest confidence
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)

    # Calculate IOUs between each pair of boxes
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    # Non-maximum suppression and render image
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            tl, width, height, confidence, class_index = bboxes[i]

            # Decrease confidence of other conflicting bboxes
            for j in range(num_boxes):
                other_class = bboxes[j][4]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Annotate image
            draw.rectangle((tl, (tl[0] + width, tl[1] + height)), outline='orange')
            text_pos = (max(0, tl[0]), max(0, tl[1] - 11))
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            text_bbox = draw.textbbox(text_pos, text)
            draw.rectangle(text_bbox, fill='orange')
            draw.text(text_pos, text)
    if file is None:
        image.show()
    else:
        if not os.path.exists(file):
            os.makedirs(file)
        # if not file.endswith('.png'):
        img_save_path = os.path.join(file, f"modified_{image_name}")
        print(img_save_path)
        image.save(img_save_path)
# def mean_avg_precision(bboxes_preds, bboxes_targets, iou_threshold = 0.5, 
#                         boxformat ="midpoints", num_classes = 20):
#     """
#     Calculates mean average precision, by collecting predicted bounding boxes on the
#     test set and then evaluate whether predictied boxes are TP or FP. Prediction with an 
#     IOU larger than 0.5 are TP and predictions larger than 0.5 are FP. Since there can be
#     more than a single bounding box for an object, TP and FP are ordered by their confidence
#     score or class probability in descending order, where the precision is computed as
#     precision = (TP / (TP + FP)) and recall is computed as recall = (TP /(TP + FN)).

#     Input: Predicted bounding boxes (list): [training index, class prediction C,
#                                               probability score p, x1, y1, x2, y2], ,[...]
#             Target/True bounding boxes:
#     Output: Mean average precision (float)
#     """

#     avg_precision = []
    
#     # iterate over classes category
#     for c in range(num_classes):
#         # init candidate detections and ground truth as an empty list for storage
#         candidate_detections = []
#         ground_truths = []
        
#         # iterate over candidate bouding box predictions 
#         for detection in bboxes_preds:
#             # index 1 is the class prediction and if equal to class c we are currently
#             # looking at append
#             # if the candidate detection in the bounding box predictions is equal 
#             # to the class category c we are currently looking at add it to 
#             # candidate list 
#             if detection[1] == c:
#                 candidate_detections.append(detection)
                
#         # iterate over true bouding boxes in the target bounding boxes
#         for true_bbox in bboxes_targets:
#             # if true box equal class category c we are currently looking at
#             # append the ground truth list
#             if true_bbox[1] == c:
#                 ground_truths.append(true_bbox)
        
#         # first index 0 is the training index, given image zero with 3 bbox
#         # and img 1 has 5 bounding boxes, Counter will count how many bboxes
#         # and create a dictionary, so amoung_bbox = [0:3, 1:5]
#         amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
#         for key, val in amount_bboxes.items():
#             # fills dic with torch tensor zeors of len num_bboxes
#             amount_bboxes[key] = torch.zeros(val)
            
#         # sort over probability scores
#         candidate_detections.sort(key=lambda x: x[2], reverse = True)
        
#         # length for true positives and false positives for class based on detection
#         # initalise tensors of zeros for true positives (TP) and false positives 
#         # (FP) as the length of possible candidate detections for a given class C
#         TP = torch.zeros((len(candidate_detections)))
#         FP = torch.zeros((len(candidate_detections)))
#         total_true_bboxes = len(ground_truths)
        
#         if total_true_bboxes == 0:
#             continue
        
#         for detection_idx, detection in enumerate(candidate_detections):
#             ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            
#             num_gts = len(ground_truth_img)
#             best_iou = 0
            
#             # iterate over all ground truth bbox in grout truth image
#             for idx, gt in enumerate(ground_truth_img):
#                 iou = intersec_over_union(
#                     # extract x1,x2,y1,y2 using index 3:
#                     bboxes_preds = torch.unsqueeze(torch.tensor(detection[3:]),0),
#                     bboxes_targets = torch.unsqueeze(torch.tensor(gt[3:]),0),
#                     boxformat = boxformat)
            
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_gt_idx = idx
                
                
#             if best_iou > iou_threshold:
#                 # check if the bounding box has already been covered or examined before
#                 if amount_bboxes[detection[0]][best_gt_idx] == 0:
#                     TP[detection_idx] = 1
#                     # set it to 1 since we already covered the bounding box
#                     amount_bboxes[detection[0]][best_gt_idx] = 1
#                 else:
#                     # if bounding box already covered previously set as FP
#                     FP[detection_idx] = 1
#             # if the iou was not greater than the treshhold set as FP
#             else:
#                 FP[detection_idx] = 1
    
#         # compute cumulative sum of true positives (TP) and false positives (FP)
#         # i.e. given [1, 1, 0, 1, 0] the cumulative sum is [1, 2, 2, 3, 3]
#         TP_cumsum = torch.cumsum(TP, dim = 0)
#         FP_cumsum = torch.cumsum(FP, dim = 0)
#         recall = torch.div(TP_cumsum , (total_true_bboxes + 1e-6))
#         precision = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
        
#         # compute average precision by integrating using numeric integration
#         # with the trapozoid method starting at point x = 1, y = 0 
#         # starting points are added to precision = x and recall = y using
#         # torch cat
#         precision = torch.cat((torch.tensor([1]), precision))
#         recall = torch.cat((torch.tensor([0]), recall))
#         integral = torch.trapz(precision, recall)
#         avg_precision.append(integral)
#         print(avg_precision)
#     return sum(avg_precision) / len(avg_precision)

def intersec_over_union(bboxes_preds, bboxes_targets, boxformat = "midpoints"):    
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

def convert_yolo_pred_x1y1x2y2(yolo_pred, S, B, C, use_sigmoid=False):
    r"""
    Method converts yolo predictions to
    x1y1x2y2 format
    """
    out = yolo_pred.reshape((S, S, 5 * B + C))
    if use_sigmoid:
        out[..., :5 * B] = torch.nn.functional.sigmoid(out[..., :5 * B])
    out = torch.clamp(out, min=0., max=1.)
    class_score, class_idx = torch.max(out[..., 5 * B:], dim=-1)

    # Create a grid using these shifts
    # Will use these for converting x_center_offset/y_center_offset
    # values to x1/y1/x2/y2(normalized 0-1)
    # S cells = 1 => each cell adds 1/S pixels of shift
    shifts_x = torch.arange(0, S, dtype=torch.int32, device=out.device) * 1 / float(S)
    shifts_y = torch.arange(0, S, dtype=torch.int32, device=out.device) * 1 / float(S)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

    boxes = []
    confidences = []
    labels = []
    for box_idx in range(B):
        # xc_offset yc_offset w h -> x1 y1 x2 y2
        boxes_x1 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) -
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y1 = ((out[..., 1 + box_idx * 5] * 1 / float(S) + shifts_y) -
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes_x2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) +
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_y) +
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes.append(torch.cat([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=-1))
        confidences.append((out[..., 4 + box_idx * 5] * class_score).reshape(-1))
        labels.append(class_idx.reshape(-1))
    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(confidences, dim=0)
    labels = torch.cat(labels, dim=0)
    return boxes, scores, labels

def non_max_suppression(bboxes, iou_threshold, threshold, boxformat="corners"):
    """
    Does Non Max Suppression given bboxes.
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersec_over_union(
                torch.tensor(chosen_box[2:], device=device),
                torch.tensor(box[2:], device=device),
                boxformat=boxformat,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def confidence_scores_threshold(boxes,conf_threshold = 0.3):
    """
    boxes:
        boxes: boxes[0]
        scores: boxes[1]
        labels: boxes[2]
    """

    boxes, scores,labels = boxes
    keep = torch.where(scores >= conf_threshold)[0]
    return boxes[keep],scores[keep],labels[keep]

def non_maximum_suppression(boxes, scores,labels, iou_threshold = 0.3):
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(labels):
        curr_indices = torch.where(labels == class_id)[0]
        curr_keep_indices = torch.ops.torchvision.nms(boxes[curr_indices],
                                                        scores[curr_indices],
                                                        iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep = torch.where(keep_mask)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    return boxes,scores,labels

def convert_pred_to_certain_format(boxes, scores,labels):
    """
    convert prediction info into the format as of [xmin, ymin, xmax, ymax, class_id, confidence]
    """
    boxes *= torch.tensor([config.IMAGE_SIZE[0],config.IMAGE_SIZE[1],config.IMAGE_SIZE[0],config.IMAGE_SIZE[1]], 
        dtype = torch.float32,device=device).expand_as(boxes)
    result = []
    for idx, box in enumerate(boxes):
        x1,y1,x2,y2 = box.detach().cpu().numpy()  # can't convert cuda:0 device type tensor to numpy. Convert to cpu first
        confidence = scores[idx].detach().item()
        cls = labels[idx].detach().item()
        result.append([x1,y1,x2,y2,cls,confidence])

    return np.array(result)

def convert_target_to_certain_format(boxes,labels,difficult):
    """
    convert target info into the format as of [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    """
    result = []
    boxes *= torch.tensor([config.IMAGE_SIZE[0],config.IMAGE_SIZE[1],config.IMAGE_SIZE[0],config.IMAGE_SIZE[1]], 
                          dtype = torch.float32, device=device).expand_as(boxes)
    for idx, box in enumerate(boxes):
        x1,y1,x2,y2 = box.detach().cpu().numpy()  # can't convert cuda:0 device type tensor to numpy. Convert to cpu first
        #confidence = scores[idx].detach().to(device).item()
        cls = labels[idx].detach().item()
        difficult = 0
        crowd = 0
        result.append([x1,y1,x2,y2,cls,difficult, crowd])
    
    return np.array(result)

def mean_average_precision(pred,conf_threshold = 0.3,iou_threshold = 0.3):
    """
    pred: YOLO output [batch_size, S,S, B*5 + C]
    target:YOLO output [batch_size, S,S, B*5 + C]
    """

    # convert yolo output 
    pred_boxes = convert_yolo_pred_x1y1x2y2(pred, config.S, config.B, config.C, use_sigmoid=False)
    #target_boxes = convert_yolo_pred_x1y1x2y2(target, config.S, config.B, config.C, use_sigmoid=False)


    # keep = torch.where(scores > conf_threshold)[0]
    # boxes = boxes[keep]
    pred_boxes, pred_scores,pred_labels = confidence_scores_threshold(pred_boxes,conf_threshold = 0.3)
    #target_boxes, target_scores,target_labels = confidence_scores_threshold(target_boxes,conf_threshold = 1)

    # non-maximum suppression 非极大值抑制
    # keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    # for class_id in torch.unique(labels):
    #     curr_indices = torch.where(labels == class_id)[0]
    #     curr_keep_indices = torch.ops.torchvision.nms(boxes[curr_indices],
    #                                                     scores[curr_indices],
    #                                                     iou_threshold)
    #     keep_mask[curr_indices[curr_keep_indices]] = True
    # keep = torch.where(keep_mask)[0]
    # boxes = boxes[keep]
    # scores = scores[keep]
    # labels = labels[keep]
    pred_boxes, pred_scores,pred_labels = non_maximum_suppression(pred_boxes, pred_scores,pred_labels, iou_threshold = 0.3)

    # convert to bb box format to 
    # pred: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    # tg: [xmin, ymin, xmax, ymax, class_id, confidence]
    # pred_boxes = []
    # target_boxes = []

    # for idx, box in enumerate(boxes):
    #     x1,y1,x2,y2 = box.detach().to(device).numpy()
    #     confidence = scores[idx].detach().to(device).item()
    #     cls = labels[idx].detach().item()
    #     difficult = 0
    #     crowd = 0
    #     pred_boxes.append([x1,y1,x2,y2,cls,difficult, crowd])
    #     pred = np.array(pred_boxes)
    preds = convert_pred_to_certain_format(pred_boxes, pred_scores,pred_labels)
    #tgs = convert_to_certain_format(target_boxes, target_scores,target_labels, pred = False)
    # print('preds',preds)
    # print('preds.shape',preds.shape)
    # print('tgs',tgs)
    return preds

def collate_function(data):
    return list(zip(*data))

# def load_model_and_dataset(model):
#     # load dataset
#     test_dataset = VOCDataset(is_train = False)
#     test_dataloader = DataLoader(test_dataset, shuffle = False, batch_size = 1, collate_fn = collate_function)

def evalute_map(test_dataloader,model,conf_threshold = 0.4, iou_threshold = 0.5):
    """
    test_dataloader needs to be batch_size = 1
    """

    AP = []
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=config.C)

    for idx, (data,targets, filename) in enumerate(test_dataloader):
        yolo_targets = torch.cat([
            target['yolo_targets'].unsqueeze(0).float().to(device)
            for target in targets], dim=0)
                
        im = torch.cat([im.unsqueeze(0).float().to(device) for im in data], dim=0)
        pred = model(im)

        bboxes = targets[0]['bboxes'].float().to(device)
        labels = targets[0]['labels'].float().to(device)
        difficult = targets[0]['difficult'].float().to(device)
        adjusted_bboxes = targets[0]['adjusted_bboxes'].float().to(device)

        pred = mean_average_precision(pred,conf_threshold,iou_threshold)
        tgs = convert_target_to_certain_format(adjusted_bboxes, labels,difficult)
        
        temp_AP = metric_fn.value(iou_threshold, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
        AP.append(temp_AP)
        #print(AP)
    AP = np.array(AP)
    mAP = AP.mean(axis = 0)
    return mAP