import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from os import listdir
import keras
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import cv2
import tqdm
import time

class HiddenPatchAnalysis:
    
    def __init__(self, img_path, IMG_SIZE, STRIDE, PATCH_SIZE, model):
        self.IMG_SIZE = IMG_SIZE
        self.STRIDE = STRIDE
        self.PATCH_SIZE = PATCH_SIZE
        self.original = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (self.IMG_SIZE, self.IMG_SIZE)),
                                     cv2.COLOR_RGB2BGR)
        self.model = model
        self.mask = None
        self.labels = None
        self.colors = None
        self.res_dict = None
        self.id = str(time.time())
        
    def hide_patch(self, patch_color = [150, 150, 150], patch_position = None):
        assert self.PATCH_SIZE%2==1, 'patch_size must be an odd number'
        hidden = self.original.copy()
        if patch_position is None:
            px = np.random.choice(self.IMG_SIZE)
            py = np.random.choice(self.IMG_SIZE)
        else:
            px = patch_position[0]
            py = patch_position[1]
        for i in range(px - self.PATCH_SIZE//2, px + self.PATCH_SIZE//2 + 1):
            for j in range(py - self.PATCH_SIZE//2, py + self.PATCH_SIZE//2 + 1):
                if i>=0 and i<self.IMG_SIZE and j>=0 and j<self.IMG_SIZE:
                    hidden[i, j] = patch_color
        return hidden, [px, py]

    def generate_hidden_images(self):
        hidden_images = {}
        for x in tqdm.tqdm(range(0, self.IMG_SIZE, self.STRIDE)):
            for y in range(0, self.IMG_SIZE, self.STRIDE):
                hidden, _ = self.hide_patch([150, 150, 150], patch_position = [x, y])
                hidden_images[(x, y)] = hidden
        return hidden_images
    
    def make_predictions(self, hidden_images):
        n = len(hidden_images)
        flatten = []
        for k in hidden_images.keys():
            flatten.append(hidden_images[k])
        flatten.append(self.original) # the last one is the original one
        flatten = np.array(flatten)
        to_recognize = tf.convert_to_tensor(flatten)
        res_hidden = self.model.predict(to_recognize, steps = n//32, verbose = 1)
        preds = decode_predictions(res_hidden, top=5)
        return preds
    
    def compute_plot_from_predictions(self):
        print('Generating hidden images')
        hidden_images = self.generate_hidden_images()
        print('Feeding the model')
        preds = self.make_predictions(hidden_images)
        print('Results collected')
        self.res_dict = {}
        k = 0
        for i in range(0, self.IMG_SIZE, self.STRIDE):
            for j in range(0, self.IMG_SIZE, self.STRIDE):
                self.res_dict[(i, j)] = preds[k]
                k+=1
        self.res_dict[(-1, -1)] = preds[-1]
        self.labels = []
        for i in self.res_dict.keys():
            if self.res_dict[i][0][1] not in self.labels:
                self.labels.append(self.res_dict[i][0][1])
        self.colors = 255 * np.random.rand(len(self.labels), 3)
        self.mask = self.original * 0
        for px in range(0, self.IMG_SIZE, self.STRIDE):
            for py in range(0, self.IMG_SIZE, self.STRIDE):
                c = self.colors[self.labels.index( self.res_dict[(px, py)][0][1],0)]
                for i in range(px - self.PATCH_SIZE//2, px + self.PATCH_SIZE//2 + 1):
                    for j in range(py - self.PATCH_SIZE//2, py + self.PATCH_SIZE//2 + 1):
                        if i>=0 and i<self.IMG_SIZE and j>=0 and j<self.IMG_SIZE:
                            self.mask[i, j] = c
                            
    def plot_mask(self, save = False):
        plt.figure(figsize = (15, 10))
        plt.subplot(221)
        plt.imshow(self.original)
        plt.subplot(222)
        plt.imshow(self.mask)
        plt.savefig(self.id+'_mask.png')
        
    def plot_class(self, some_class, save=False):
        hm = np.zeros((self.IMG_SIZE, self.IMG_SIZE))
        for px in range(0, self.IMG_SIZE, self.STRIDE):
            for py in range(0, self.IMG_SIZE, self.STRIDE):
                scores = self.res_dict[(px, py)]
                for k in range(len(scores)):
                    if scores[k][1] == some_class:
                        for i in range(px - self.PATCH_SIZE//2, px + self.PATCH_SIZE//2 + 1):
                            for j in range(py - self.PATCH_SIZE//2, py + self.PATCH_SIZE//2 + 1):
                                if i>=0 and i<self.IMG_SIZE and j>=0 and j<self.IMG_SIZE:
                                    hm[i, j] = scores[k][2]
        plt.figure(figsize = (15, 10))
        plt.subplot(221)
        plt.imshow(self.original)
        plt.subplot(222)
        h = plt.imshow(hm, cmap = plt.cm.YlOrRd_r)
        plt.colorbar(h)
        plt.savefig(self.id+'_class.png')
        
    def plot_legend(self):
        for i in range(len(self.labels)):
            plt.figure(figsize = (3, 3))
            blank_image = np.zeros((2, 5, 3), np.uint8)
            for x in range(2):
                for y in range(5):
                    blank_image[x, y, :] = self.colors[i]
            plt.imshow(blank_image)
            plt.title(self.labels[i])