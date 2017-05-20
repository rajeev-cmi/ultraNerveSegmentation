from __future__ import print_function
#import sys
#sys.setrecursionlimit(10000)

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import os
import nibabel as nib
from pdb import set_trace as trace
import math
import json
from datetime import datetime
from skimage.io import imread
import theano
import theano.tensor as T
import resnetModel
from pdb import set_trace as trace

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


img_rows = 224
img_cols = 224
chunk_size = 64
chunk_repeat = 16
channels = 1
smooth = 1.



def chunks(l1, l2, n, k):
    for i in range(0, len(l1), n-k):
        yield (l1[i:i+n], l2[i:i+n])



def preprocess(imgs):
    dim = imgs.shape
    
    tmp = np.zeros((dim[0],1,dim[1],dim[2]))
    for i in range(dim[0]):
        tmp[i] = imgs[i,:,:]
    imgs = tmp
    #imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    #for i in range(imgs.shape[0]):
    #    imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    #dim = imgs_p.shape

    #return imgs_p.reshape(dim[0],dim[2],dim[3])
    return imgs


def train_and_predict():
    
    model = resnetModel.ResnetModel(in_channels=1, classes=1)
    
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)


    #imgs_train, imgs_mask_train = load_train_data()
    data_path = '/home/deepak/Desktop/ultrasound-nerve-segmentation-master/raw/'
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)

    total = len(images) / 2

    imgs_train = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_mask_train = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        #print(image_name)
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array(cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC))
        img_mask = np.array(cv2.resize(img_mask, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC))

        imgs_train[i] = img
        imgs_mask_train[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print('-'*30)
    print('Preprocessing now')
    imgs_train = preprocess(imgs_train)
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = cv2.threshold(imgs_mask_train,0.5,1,cv2.THRESH_BINARY)[1]
    imgs_mask_train = preprocess(imgs_mask_train)
    print('Preprocessing done.')
    print('-'*30)
    #trace()
    print('Fitting model...')
    print('-'*30)

    num_epochs = 20          
    model.fit(imgs_train, imgs_mask_train, batch_size=10, epochs=num_epochs, verbose=1)
    #print(e+1)

    # serialize model to JSON
    print("Model Building is complete, Save Model to Disk...")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    now = datetime.now()
    train_and_predict()
    print('Total Time Taken:',datetime.now()-now)
