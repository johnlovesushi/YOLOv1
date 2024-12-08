import torch
import json
import os
import config
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
from collections import Counter
import numpy as np

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

    tl = torch.max(                                  # top left
        p_tl.unsqueeze(4).expand(coords_join_size),  # (batch,S,S,B,1,2) -> (batch,S,S,B,2,2)
        a_tl.unsqueeze(3).expand(coords_join_size)   # (batch, S,S,1,B,2) -> (batch, S,S,1,B,2)
    )
    br = torch.min(                                  # bottom right
        p_br.unsqueeze(4).expand(coords_join_size),  # (batch,S,S,B,1,2) -> (batch,S,S,B,2,2)
        a_br.unsqueeze(3).expand(coords_join_size)   # (batch, S,S,1,B,2) -> (batch, S,S,1,B,2)
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                * intersection_sides[..., 1]       # (batch, S, S, B, B)


    p_area = bbox_attr(p, 3) * bbox_attr(p, 4)  # (batch,S,S,B)         w * h
    a_area = bbox_attr(a, 3) * bbox_attr(a, 4)  # (batch,S,S,B)

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

    width = bbox_attr(t, 3)
    x = bbox_attr(t, 1)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 4)
    y = bbox_attr(t, 2)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)

def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""
    # [classes [0: 19], conf[20], x[21],y[22], w[23], h[24]]
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
    """Returns proportion overlap between two boxes in the form (confidence, class, tl, width, height)."""

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


def plot_boxes(data, targets, classes, color='orange', min_confidence=0.5, max_overlap=0.5, file_name=None, output_path = "./output/"):
    """Plots bounding boxes on the given image based on the output from the model
    @params:
        data(Tensor): the original image data in 3 dimension ([3,448,448])
        targets(Tensor): the results generated from the model
        classes(list): the list of classes
        color(str): bounding box color, default as organe
        min_confidence(float): minimum confidence that allows the bounding boxes
        max_overlap (float): max IOU between two bounding boxes. Otherwise, the one with lower confidence will be discarded
        file_name (str): image name
        output_path (str): output path
    @return:
        none
    """

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S
    m = targets.size(dim=0)
    n = targets.size(dim=1)
    #image_name = labels['annotation']['filename']
    bboxes = []
    for i in range(m):
        for j in range(n):
            for k in range((targets.size(dim=2) - config.C) // 5):
                bbox_start = 5 * k + config.C
                bbox_end = 5 * (k + 1) + config.C
                bbox = targets[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(targets[i, j, :config.C]).item()
                confidence = targets[i, j, class_index].item() * bbox[0].item()          # pr(c) * IOU
                if confidence > min_confidence:
                    width = bbox[3] * grid_size_x
                    height = bbox[4] * grid_size_y
                    lt = (
                        bbox[1] *  grid_size_y + j * grid_size_y - height / 2,
                        bbox[2] *  grid_size_x + i * grid_size_x - width / 2 
                    )          
                    bboxes.append([lt, width, height, confidence, class_index])

    # Sort by highest to lowest confidence
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)

    # Calculate IOUs between each pair of boxes
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    print(bboxes)
    # Non-maximum suppression and render image
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            lt, width, height, confidence, class_index = bboxes[i]

            # Decrease confidence of other conflicting bboxes
            for j in range(num_boxes):
                other_class = bboxes[j][4]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Annotate image
            draw.rectangle((lt, (lt[0] + height, lt[1] + width)), outline='orange')
            text_pos = (max(0, lt[0]), max(0, lt[1] - 11))
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            text_bbox = draw.textbbox(text_pos, text)
            draw.rectangle(text_bbox, fill='orange')
            draw.text(text_pos, text)
    if file_name is None:
        image.show()
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # if not file.endswith('.png'):
        img_save_path = os.path.join(output_path, f"modified_{file_name}")
        print(img_save_path)
        image.save(img_save_path)
    
    return

def intersec_over_union(bboxes_preds, bboxes_targets, boxformat = "midpoints"):    
    """
    Calculates intersection of unions (IoU).
    @params: 
            Boundbing box predictions (tensor) x1, x2, y1, y2 of shape (N , 4)
            with N denoting the number of bounding boxes.
            Bounding box target/ground truth (tensor) x1, x2, y1, y2 of shape (N, 4).
            box format whether midpoint location or corner location of bounding boxes
            are used.
    @return: 
            Intersection over union (tensor).
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
    1.NMS sets a confidence thresholds, discarding boxes below that threshold.
    2.sorted by confidence score, and then go over the rest of bboxes and get rid of those which has a IOU greater than the threshold
    @params:
        bboxes (list): list of lists containing all bboxes with each sample
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct 
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    @returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # sorted by confidence score
    bboxes_after_nms = []

    while bboxes:
        # go over all boxes, and filter 
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

def collate_function(data):
    return list(zip(*data))

def convert_bboxes_entire_img_ratios(yolo_output, S = 7):
    """
    convert yolo output bounding boxes into the entire images ratio
    input: 
            yolo_output: [batch_size, S,S, C + 2*B]
    output:
            [batch_size,S*S,6] => [predicted_class, scores, x,y,w,h] in entire image ratio
    """
    batch_size = yolo_output.shape[0]
    bbox1, bbox2 = yolo_output[...,21:25], yolo_output[...,26:30]
    scores = torch.cat([yolo_output[..., 20].unsqueeze(0), yolo_output[..., 25].unsqueeze(0)], dim=0)   #[2,batch_size,S,S]
    best_scores = scores.argmax(0).unsqueeze(-1)                    #[batch_size,S,S,1]
    best_boxes = bbox1 * (1-best_scores) + bbox2 * best_scores      # [batch_size, S,S, 4]
    # convert x,y,w,h to entire image ratio
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1).to(device)  # [batch_size, S,S, 1]
    # [coord_y,coord_x,  20]
    print(best_boxes.device, cell_indices.device)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = yolo_output[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(yolo_output[..., 20], yolo_output[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat([predicted_class, best_confidence, converted_bboxes], dim = -1)
    return converted_preds

def convert_bboxes_to_list(yolo_output, S = 7):
    converted_bboxes = convert_bboxes_entire_img_ratios(yolo_output).reshape(yolo_output.shape[0],S*S,-1)
    batch_bboxes_lst = []
    num_batch = converted_bboxes.shape[0]
    for i in range(num_batch):
        bboxes_list = []
        for j in range(S*S):
            bboxes_list.append([x.item() for x in converted_bboxes[i,j, :]])
        batch_bboxes_lst.append(bboxes_list)
    #print("batch_bboxes_lst",batch_bboxes_lst)
    return batch_bboxes_lst

def get_bboxes(loader,model, iou_threshold = 0.5, threshold = 0.4,boxformat="midpoints",
    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    obtain the bboxes from the targets and preds after non maximum supression and supressing overlap, and organize it to the format for mAP calculation
    input:
        loader
        model
    output:
        batch_pred_bboxes: list
        batch_true_bboxes: list
    """
    batch_true_bboxes = []
    batch_pred_bboxes = []
    model.eval()

    train_idx = 0
    with torch.no_grad():
        for batch_idx, (data,targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)
            preds = model(data)

            batch_size = data.shape[0]
            preds_bboxes = convert_bboxes_to_list(preds)
            true_bboxes = convert_bboxes_to_list(targets)

            for idx in range(batch_size):
                # conduct non maximum supression for each of the sample in the batch
                nms_boxes = non_max_suppression(preds_bboxes[idx], iou_threshold=iou_threshold, threshold=threshold,boxformat=boxformat)

                for nms_box in nms_boxes:
                    batch_pred_bboxes.append([train_idx] + nms_box)

                for box in true_bboxes[idx]:
                    # many will get converted to 0 pred
                    if box[1] > threshold:  # confidence greater than threshold 
                        batch_true_bboxes.append([train_idx] + box)

            train_idx += 1
    return batch_pred_bboxes, batch_true_bboxes

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, boxformat="midpoints", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            # num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersec_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    boxformat=boxformat,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def x1y1x2y2_convert_to_xywh(bboxes_tensor):
    
    target_bbox_tensor = torch.zeros_like(bboxes_tensor)
    num_of_bboxes = bboxes_tensor.shape[0]
    for i in range(num_of_bboxes):
        xmin,ymin,xmax,ymax = bboxes_tensor[i,...]
        x_middle = (xmax + xmin)/2
        y_middle = (ymax + ymin)/2
        w = xmax-xmin
        h = ymax-ymin
        target_bbox_tensor[i,...] = torch.tensor([x_middle, y_middle, w, h])      

    return target_bbox_tensor  
