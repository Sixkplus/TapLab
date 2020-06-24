import numpy as np
import cv2
import scipy.io as sio
import torch
import torch.nn.functional as F



def set_img_color(colors, background, img, gt, show255=False):
    for i in range(1, len(colors)):
        if i != background:
            img[np.where(gt == i)] = colors[i]
    if show255:
        img[np.where(gt == 255)] = 255
    return img


def show_prediction(colors, background, img, pred):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred)
    final = np.array(im)
    return final


def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    set_img_color(colors, background, im1, clean)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt, True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final


def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1, 3)) * 255).tolist()[0])

    return colors


def get_ade_colors():
    colors = sio.loadmat('./color150.mat')['colors']
    colors = colors[:, ::-1, ]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0, [0, 0, 0])

    return colors


def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False,
              no_print=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append(
            '----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % (
                'mean_IU', mean_IU * 100, 'mean_IU_no_back',
                mean_IU_no_back * 100,
                'mean_pixel_ACC', mean_pixel_acc * 100))
    else:
        print(mean_pixel_acc)
        lines.append(
            '----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%' % (
                'mean_IU', mean_IU * 100, 'mean_pixel_ACC',
                mean_pixel_acc * 100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line


def decode_color(colors, preds, num_classes=19):
    '''
    colors:         map from id to color(rgb)
    preds:          a batch of predictions [n, c, h, w]
    num_classes:    number of classes

    return: a batch of colored predictions
    '''
    
    colors = torch.Tensor(colors).cuda().long()

    if len(preds.shape) == 4:
        # to [n,h,w,c]
        preds = preds.permute(0,2,3,1)
    # take max
    preds = preds.argmax(3)
    # one-hot
    preds = F.one_hot(preds, num_classes)
    return (preds.float() @ colors.float()).permute(0,3,1,2)
    

def de_nomalize(images, mean, std):
    '''
    de-nomalize the input image for visulization
    
    images: tensor, [n,3,h,w]
    mean:   mean for r, g, b
    std:    standard Deviation
    '''

    mean = torch.Tensor(mean).cuda()
    std = torch.Tensor(std).cuda()

    return ((images.permute(0,2,3,1) * std) + mean).permute(0,3,1,2)


label_colours_cityscapes = [[128, 64, 128], [244, 35, 231], [69, 69, 69]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 29], [219, 219, 0], [106, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 250, 152], [69, 129, 180], [219, 19, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 69]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 79, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle

label_colours_camvid = [
    # 0 = building, 1 = tree, 2 = sky
    [128, 0, 0], [128,128,0], [128,128,128], 
    # 3 = car, 4 = signsymbol, 5 = road
    [64,0,128], [192,128,128], [128,64,128], 
    # 6 = pedestrian, 7 = fence, 8 = column_pole
    [64,64,0], [64,64,128], [192,192,128], 
    # 9 = sidewalk, 10 = bicyclist
    [0,0,192], [0,128,192]
]

label_colours_freetech = [[153,153,153], [128, 64, 128], [0, 0, 142]
                # 0 = background, 1 = road, 2 = vehicle
                ,[255, 0, 0], [219, 19, 60], [219, 219, 0]]
                # 3 = rider, 4 = walker, 5 = cone

label_colours_agric = [[153,153,153], [255,0,0], [219, 219, 0]
                # 0 = background, 1 = 烤烟, 2 = 玉米
                ,[0, 0, 255]]
                # 3 = 薏米仁

id_list_cityscapes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
id_list_freetech = [0,1,2,3,4,5]
id_list_agric = [0,1,2,3]
id_list_camvid = [0,1,2,3,4,5,6,7,8,9,10]

freetech_id_list = []

def decode_labels(mask, img_shape, num_classes):
    if num_classes == 6:
        color_table = label_colours_freetech
    elif num_classes == 4:
        color_table = label_colours_agric
    elif num_classes == 11:
        color_table = label_colours_camvid
    else:
        color_table = label_colours_cityscapes

    color_mat = np.array(color_table)
    # one-hot
    one_hot_matrix = np.eye(num_classes)
    onehot_output = one_hot_matrix[mask]
    onehot_output = np.reshape(onehot_output, (-1, num_classes))

    pred = onehot_output@color_mat
    pred = np.reshape(pred, (int(img_shape[0]), int(img_shape[1]), 3)).astype(np.uint8)
    
    return pred

def decode_ids(mask, img_shape, num_classes):
    if num_classes == 6:
        id_list = id_list_freetech
    elif num_classes ==4:
        id_list = id_list_agric
    elif num_classes == 11:
        id_list = id_list_camvid
    else:
        id_list = id_list_cityscapes
    id_mat = np.array(id_list).reshape((num_classes, 1))
    # one-hot
    one_hot_matrix = np.eye(num_classes)
    onehot_output = one_hot_matrix[mask]
    onehot_output = np.reshape(onehot_output, (-1, num_classes))
    
    pred = onehot_output@id_mat
    pred = np.reshape(pred, (int(img_shape[0]), int(img_shape[1]), 1)).astype(np.uint8)
    
    return pred
