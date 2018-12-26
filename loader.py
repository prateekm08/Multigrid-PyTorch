import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import merge_images, par_imread

import os
import time
import re
import skimage.io
import random
import numpy as np
import cv2

import matplotlib.pyplot as plt

dtype = torch.FloatTensor
if torch.cuda.is_available() == True:
    dtype = torch.cuda.FloatTensor

def image_to_variable(image):
    return Variable(torch.from_numpy(image.transpose(2, 0, 1)[np.newaxis, ...]), requires_grad=True).type(dtype)

def process_data_seq(FLAGS):
    direc = FLAGS.dataset_name
    dataT = []
    pattern = FLAGS.input_pattern
    scale_list = FLAGS.scale_list

    for im_name in os.listdir(direc):
        if pattern in im_name:
            im_name = os.path.join(direc, im_name)
            image = imread_file(im_name, 64)
            dataT.append(image_to_variable(image))

    random.shuffle(dataT)
    data = []

    for image in dataT:
        data.append(scaled_down_images(image, scale_list))
    print(data.shape)
    
    return data

def process_data_parallel(FLAGS):
    direc = FLAGS.dataset_name
    names = []
    pattern = FLAGS.input_pattern
    scale_list = FLAGS.scale_list

    for im_name in os.listdir(direc):
        if pattern in im_name:
            im_name = os.path.join(direc, im_name)
            names.append(im_name)
            
    dataT = par_imread(names, 64, FLAGS.num_threads)
    for i in range(len(dataT)):
        dataT[i] = image_to_variable(dataT[i])
        
    random.shuffle(dataT)
    data = []

    for image in dataT:
        data.append(scaled_down_images(image, scale_list))
    
    return data
        
def scale_down(imageT, to_size):
    assert imageT.size(2) > to_size, "You might wanna look at scale_up"
        
    filter_size = imageT.size(2) // to_size
    return F.avg_pool2d(imageT, filter_size)

def scale_up(imageT, to_size):
    assert imageT.size(2) < to_size, "You might wanna look at scale_down"
    
    filter_size = to_size // imageT.size(2)
    
    temp_w_inv = torch.zeros([3, 3, filter_size, filter_size])
    temp_w_inv[0, 0, :, :] = 1
    temp_w_inv[1, 1, :, :] = 1
    temp_w_inv[2, 2, :, :] = 1
    
    temp_w_inv = Variable(temp_w_inv).type(dtype)
    
    return F.conv_transpose2d(imageT, temp_w_inv, stride=filter_size)

def scaled_down_images(imageT, scale_list):    #Input image of 64 x 64
    images = {}
    
    images[str(scale_list[-1])] = imageT.type(dtype)
    
    for scale in reversed(scale_list[:-1]):
        images['{}'.format(scale)] = scale_down(imageT, scale)
        
    return images