import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import merge_images, par_imread

from loader import *

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

class Multigrid(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.sample_dir = FLAGS.sample_dir
        self.learn_rate = FLAGS.learning_rate
        self.wt_decay = FLAGS.weight_decay
        self.beta1 = FLAGS.beta1
        self.load_models = FLAGS.load_models
        self.model_dir = FLAGS.model_dir
        self.delta = FLAGS.delta
        self.L = FLAGS.T
        self.scale_list = FLAGS.scale_list
        self.epoch_file = FLAGS.epoch_file
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.models = {}
        self.optimizers = {}
        
        grid1 = self.net_wrapper(4).type(dtype)
        grid2 = self.net_wrapper(16).type(dtype)
        grid3 = self.net_wrapper(64).type(dtype)

        optimizer1 = optim.Adam(grid1.parameters(), lr=self.learn_rate, betas=(self.beta1, 0.999), weight_decay=self.wt_decay)
        optimizer2 = optim.Adam(grid2.parameters(), lr=self.learn_rate, betas=(self.beta1, 0.999), weight_decay=self.wt_decay)
        optimizer3 = optim.Adam(grid3.parameters(), lr=self.learn_rate, betas=(self.beta1, 0.999), weight_decay=self.wt_decay)

        if self.load_models is False:
            grid1.apply(self.weights_init)
            grid2.apply(self.weights_init)
            grid3.apply(self.weights_init)

        else:
            load_dir = self.model_dir

            grid1.load_state_dict(torch.load(os.path.join(load_dir, 'model_4.pt')))
            grid2.load_state_dict(torch.load(os.path.join(load_dir, 'model_16.pt')))
            grid3.load_state_dict(torch.load(os.path.join(load_dir, 'model_64.pt')))
            optimizer1.load_state_dict(torch.load(os.path.join(load_dir, 'optimizer_4.pt')))
            optimizer2.load_state_dict(torch.load(os.path.join(load_dir, 'optimizer_16.pt')))
            optimizer3.load_state_dict(torch.load(os.path.join(load_dir, 'optimizer_64.pt')))

        self.models = {'4':grid1, '16':grid2, '64':grid3}
        self.optimizers = {'4':optimizer1, '16':optimizer2, '64':optimizer3}

    def net_wrapper(self, im_size):
        if im_size == 4:
            return net1().type(dtype)
        elif im_size == 16:
            return net2().type(dtype)
        elif im_size == 64:
            return net3().type(dtype)
        else:
            print('Error!! unsupported model version {}'.format(im_size))
            exit()

    def weights_init(self, m):                           #Weight initialization
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)

    def langevin_dynamics(self, samples, size):
        noise = torch.zeros((1, samples.size(1), size, size)).type(dtype)

        self.models[str(size)].train()
        samples = scale_up(samples, size)
        samples.retain_grad()
        delta = self.delta

        for l in range(self.L):
            noise_term = (np.sqrt(delta) * noise.normal_(0, 1)).type(dtype)
            E = self.models[str(size)](samples)
            ones = torch.ones(E.size()).type(dtype)
            E.backward(ones, retain_graph=True)

            samples.data = torch.clamp(samples.data + 0.5 * delta * delta * samples.grad.data, min=0, max=255) + noise_term
            samples.grad.data.zero_()

        return samples

    def hundredimages(self, data): #Generates 100 sample full-size images from 1x1 pixel values for displaying
        
        originals = Variable(torch.zeros((100, 3, 1, 1))).type(dtype)
        scale_list = self.scale_list
        
        for i in range(100):
            rand = np.random.randint(1000)
            originals[i] = data[rand]['1']

        samples = {str(scale):[] for scale in scale_list}
        samples['1'] = originals.data
        samples['1'] = Variable(samples['1'], requires_grad=True).type(dtype)

        for i in range(1, len(scale_list)):
            samples[str(scale_list[i])] = self.langevin_dynamics(samples[str(scale_list[i-1])], scale_list[i])

        return samples

    def train(self):

        data = process_data_parallel(self.FLAGS)
        file = open(self.epoch_file, "r")
        e = int(file.read())
        file.close()
        epochs = self.epochs - e            #Epochs
        print('Will train for {} epochs'.format(epochs))

        batch_size = self.batch_size
        scale_list = self.scale_list

        #Step 0: Put models in training mode
        for size, model in self.models.items():
            model.train()

        #Step 1: Traverse all images E (number of epochs) times
        for epoch in range(epochs):
            start_time = time.time()

            #Step 2: Create mini-batches to traverse
            for batch in range(len(data) // batch_size):

                batch_ori = {str(scale):[] for scale in scale_list}             #Originals
                samples = {str(scale):[] for scale in scale_list}

                #Step 3: Traverse all images in the mini-batch
                for image_no in range(batch * batch_size, min((batch + 1) * batch_size, len(data))):

                    #Extract images from each traversed image and store them in a list
                    for scale in scale_list:
                        batch_ori[str(scale)].append(data[image_no][str(scale)])

                #Step 4: Convert original list of images into Tensors
                for scale in scale_list:
                    batch_ori[str(scale)] = torch.cat(batch_ori[str(scale)])

                samples['1'] = batch_ori['1']

                #Step 5: L steps of Langevin Dynamics to get generated samples of different sizes
                for i in range(1, len(scale_list)):
                    samples[str(scale_list[i])] = self.langevin_dynamics(samples[str(scale_list[i-1])], scale_list[i])

                #Step 6: Compute the loss and update network parameters
                losses = {}

                for scale in scale_list[1:]:
                    E_samples = torch.mean(self.models[str(scale)](samples[str(scale)]))
                    E_ori = torch.mean(self.models[str(scale)](batch_ori[str(scale)]))                
                    losses[str(scale)] = E_samples - E_ori
                    reconstruction_loss = ((E_samples - E_ori)**2).mean()

                    self.optimizers[str(scale)].zero_grad()
                    losses[str(scale)].backward(retain_graph=True)

                    self.optimizers[str(scale)].step()

            #Step 7: Save the models after every epoch
            load_dir = self.model_dir

            for size, model in self.models.items():
                torch.save(model.state_dict(), os.path.join(load_dir, 'model_{}.pt'.format(size)))

            #Step 8: Save the state of the optimizers after every epoch
            for size, optimizer in self.optimizers.items():
                torch.save(optimizer.state_dict(), os.path.join(load_dir, 'optimizer_{}.pt'.format(size)))

            #Step 9: Save number of completed epochs in a text file
            file = open(self.epoch_file, "r")
            epo = int(file.read())
            file.close()

            file = open(self.epoch_file, "w")
            file.write(str(epo + 1))
            file.close()

            #Step 10: Save the results generated
            himages = self.hundredimages(data)['64'].data.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
            canvas = merge_images(himages)
            skimage.io.imsave(os.path.join(self.sample_dir, 'epoch{}.jpg'.format(epo + 1)), canvas)

            print('Epoch {} completed; Time taken: {}; Reconstruction Loss: {}'.format(epo + 1, time.time() - start_time, 
                                                                                        reconstruction_loss.data.cpu()[0]))
            
            
class net1(nn.Module):        #4x4 images
    def __init__(self):
        super(net1, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(96)                                                #2x2 now

        self.conv2 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)                                               #2x2 now

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)         
        self.bn3 = nn.BatchNorm2d(256)                                               #2x2 now

        self.dense = nn.Linear(2 * 2 * 256, 1)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        return self.dense(x)

class net2(nn.Module):    #16x16 images
    def __init__(self):
        super(net2, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(96)                                              #8x8 now

        self.conv2 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)                                             #8x8 now

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)                                             #8x8 now

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)                                             #8x8 now

        self.dense = nn.Linear(8 * 8 * 512, 1)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        return self.dense(x)

class net3(nn.Module):  #64x64 images
    def __init__(self):
        super(net3, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(96)                                               #32x32 now

        self.conv2 = nn.Conv2d(96, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)                                              #16x16 now       

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)                                              #8x8 now

        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)                                              #4x4 now

        self.dense = nn.Linear(4 * 4 * 512, 1)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        return self.dense(x)
