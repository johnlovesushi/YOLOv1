import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def scale_image(image, factor = 20): 
    """
    Input: PIL.Image.Image.
           scale_img (boolean) determining wheter to scale the image.
           translate (boolean) determining wheter to translate the image.
           factoor (int) how much to translate and/or scale the image with
           respect to the images original size. Default 20 is 20 %.
    Output: transformed image.
    """
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    
    # Scaling variables
    x_up_bound = width
    x_low_bound = width - (width / 100 * factor)
    x_scale_to = np.random.randint(low = x_low_bound, high = x_up_bound)

    # x_scale_to means randomly choose a value in between 0.8* wdith and width
    y_up_bound = height
    y_low_bound = height - (height / 100 * factor)
    y_scale_to = np.random.randint(low = y_low_bound, high = y_up_bound)
    #x_scale_to, y_scale_to = int(x_low_bound),int(y_low_bound)
    # print(x_scale_to,y_scale_to)  # 缩图后比例
    # y_scale_to means randomly choose a value in between 0.8* wdith and width
    x_ratio_percentage = x_scale_to / width * 100
    y_ratio_percentage = y_scale_to / height * 100
    # 原图缩小比例
    # calculate the ratio, should be 0 - 1 value 

    # Translation variables
    x_upper_bound = float(width / 100 * factor) # 正负20%的比例的值，是原图的20%的比例
    x_lower_bound = float(width / 100 * factor) * -1
    y_upper_bound = float(height / 100 * factor) 
    y_lower_bound = float(height / 100 * factor) * -1
    
    # Uniform vals to translate into x coord t_x and y coord t_y
    t_x = np.random.uniform(low = x_lower_bound, high = x_upper_bound)
    t_y = np.random.uniform(low = y_lower_bound, high = y_upper_bound)
    #t_x,t_y = 15,15
    # Translation matrix T
    T = np.float32([[1, 0, t_x], [0, 1, t_y]])
    
    # Scale image
    scaled_img = cv.resize(image, (x_scale_to, y_scale_to), interpolation = cv.INTER_CUBIC) # 图片缩放新坐标
    height_scaled, width_scaled = scaled_img.shape[:2] # 新图的高度和宽度
    blankimg = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    height_blank, width_blank = blankimg.shape[:2] # 原图坐标
    # print('height_blank, width_blank',height_blank, width_blank)
    yoff = round((height_blank - height_scaled) / 2)
    xoff = round((width_blank - width_scaled) / 2)

    result = blankimg.copy()
    result[yoff:yoff + height_scaled, xoff:xoff + width_scaled] = scaled_img
    scaled_img = result

    # Translate image after performing scaling
    img_scale_trans = cv.warpAffine(scaled_img, T, (width, height))
    #print('img_scale_trans', img_scale_trans.shape)
    # Convert from opencv format to pil
    # Opencv uses brg, pil uses rgb: convert brg to rgb
    img_scale_trans = cv.cvtColor(img_scale_trans, cv.COLOR_BGR2RGB)
    img_scale_trans = Image.fromarray(img_scale_trans)

    transform_vals = np.array([[int(height), int(width)],
                               [t_x, t_y], 
                                [xoff, yoff],
                                [x_ratio_percentage, 
                                 y_ratio_percentage]])
    
    return img_scale_trans, transform_vals

def scale_translate_bounding_box(bounding_boxes, trans_vals):
    """
    bounding_boxes: in x,y,w,h format
    """
    t_x, t_y  = trans_vals[1]
    xoff, yoff = trans_vals[2]
    height, width = trans_vals[0]
    x_ratio_percentage, y_ratio_percentage = trans_vals[3]
    transformed_bounding_boxes = torch.zeros_like(bounding_boxes)
    num_bbox = bounding_boxes.shape[0]
    for i in range(num_bbox):
        x,y,w,h = bounding_boxes[i,...]
        scaled_x = np.clip(x*x_ratio_percentage/100 + (xoff / width) + (t_x / width), 0, 0.999)
        scaled_y = np.clip(y*y_ratio_percentage/100 + (yoff / height) + (t_y / height), 0, 0.999)
        scaled_w =  np.clip( ((w / 100) * x_ratio_percentage), 0, 0.999)
        scaled_h =  np.clip( ((h/ 100) * y_ratio_percentage), 0, 0.999)    
        transformed_bounding_boxes[i,...] = torch.tensor([scaled_x,scaled_y,scaled_w,scaled_h])
    return transformed_bounding_boxes


def test():
    image = cv.imread(r'C:\Users\linzi\OneDrive\Documents\GitHub\YOLOv1\data\VOCdevkit\VOC2007\JPEGImages\000017.jpg')
    print(image.shape)

if __name__ == '__main__':
    test()