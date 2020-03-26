import os
from os.path import isdir, exists, abspath, join
import random
import numpy as np
from PIL import Image

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1): 
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels') 

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            
            # todo: load images and labels
            data_image_orig = Image.open(self.data_files[current])
            label_image_orig = Image.open(self.label_files[current])
            
            # Resizing
            data_image_orig = data_image_orig.resize((572,572))
            # To crop change to 572 and un comment next line
            # To not crop 388 (check assignment chart again)
            #label_image_orig = label_image_orig.resize((388,388)) 
            label_image_orig = label_image_orig.resize((572,572)) 
            label_image_orig = label_image_orig.crop((92, 92, 480, 480))
            
            # hint: scale images between 0 and 1
            data_image = np.asarray(data_image_orig)
            data_image = data_image/255.0
            
            label_image = np.asarray(label_image_orig)
            
            #Moved current update down, as it was causing index error at the end of the dataset (38)
            current += 1

            yield (data_image, label_image)
            
            
            ## AUGMENTATION ##
            # Rotation by 90  degree
            data_image = data_image_orig.rotate(90)
            label_image = label_image_orig.rotate(90)
            data_image = np.asarray(data_image)
            data_image = data_image/255.0
            label_image = np.asarray(label_image)
            yield (data_image, label_image)
               
            # Flip left right
            data_image = data_image_orig.transpose(Image.FLIP_LEFT_RIGHT)
            label_image = label_image_orig.transpose(Image.FLIP_LEFT_RIGHT)
            data_image = np.asarray(data_image)
            data_image = data_image/255.0
            label_image = np.asarray(label_image)
            yield (data_image, label_image)
            
            # Gamma correction with gamma=1.7
            data_image = data_image_orig
            label_image = label_image_orig
            data_image = np.asarray(data_image)
            gamma=1.7
            gamma_inv = 1.0/gamma
            data_image = (data_image/255.0)**(gamma_inv) #*255.0?
            label_image = np.asarray(label_image)
            yield (data_image, label_image)
            

    def setMode(self, mode):
        self.mode = mode
        
    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))