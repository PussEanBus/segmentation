from flask import Flask
from flask_cors import CORS
from model import SiNet
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from keras.layers import (
    Conv2D, BatchNormalization, Dense, 
    ZeroPadding2D, Activation, GlobalAveragePooling2D,
    Reshape, Permute, multiply, AveragePooling2D,
    UpSampling2D, Concatenate, Add, Lambda, Multiply
)
from keras.models import Model, Sequential
from keras.layers import Input
import keras.backend as K
from keras.layers import DepthwiseConv2D, PReLU
import cv2
from glob import glob
import pandas as pd
from sklearn.utils import shuffle
import imgaug as ia
from imgaug import augmenters as iaa
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, BaseLogger
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
CLASSES = ["Background", "Person"]
N_CLASSES = 2
IMAGE_DATA_FORMAT = K.image_data_format()

app = Flask(__name__)
CORS(app)
K.clear_session()
sinet = SiNet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL, N_CLASSES)
model = sinet.build_decoder()
model.load_weights("weights/best_weights_4_all.h5")

@app.route("/predict")
def predict():
    img_origin = cv2.imread("image_test/1.png")
    img_resize = cv2.resize(img_origin, (224, 224))[..., ::-1]
    img = np.expand_dims(img_resize, axis=0)
    prediction = model.predict(img)
    prediction = prediction[0]
    print(prediction)
    mask = np.reshape(prediction, (IMG_HEIGHT, IMG_WIDTH, N_CLASSES))
    mask = np.argmax(mask, axis=-1)
    mask[mask > 0] = 255
    
    new_img = np.copy(img_resize)
    non_zeros_idx = np.where(mask == 0)
#     non_zeros_idx = np.nonzero()
    new_img[..., 0][non_zeros_idx] = 0
    new_img[..., 1][non_zeros_idx] = 0
    new_img[..., 2][non_zeros_idx] = 0
    
#     print(mask)
#     for i in range(N_CLASSES):
#     class_idx = prediction.argmax(axis=-1)
    mask = cv2.merge([mask, mask, mask])
    
    img_resize = cv2.resize(img_resize, (512, 512))
    new_img = cv2.resize(new_img, (512, 512))
    
    cv2.imshow('',new_img)
	# return "Hello world"



if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
