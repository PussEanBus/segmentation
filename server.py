import os
import sys
import warnings

# ------------------ Flask --------------------- #

from flask import Flask
from flask_cors import CORS


# ------------------ CV --------------------- #

from model import SiNet
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
from datagenerator import DataGenerator
from dataaugentation import DataAugmentation
# ------------------ General config --------------------- #

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------ Flask config --------------------- #

HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
app = Flask(__name__)
CORS(app)

# ------------------ CV config --------------------- #

# Image config
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
CLASSES = ["Background", "Person"]
DATA_DIR = 'Nukki'
VAL_ANNO_FILE1 = os.path.join(DATA_DIR, "baidu_V1/val.txt")
VAL_ANNO_FILE2 = os.path.join(DATA_DIR, "baidu_V2/val.txt")
N_CLASSES = 2
TEST_IMAGE_PATH = os.path.join(BASE_DIR, 'image_test', '1.png')
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, 'image_test', 'output.png')

# Model
WEIGHT_FILE_PATH = os.path.join(BASE_DIR, 'weights', 'best_weights_4_all.h5')
IMAGE_DATA_FORMAT = K.image_data_format()
K.clear_session()
sinet = SiNet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL, N_CLASSES)
model = sinet.build_decoder()
model.load_weights(WEIGHT_FILE_PATH)

# data loader
data_aug = DataAugmentation()
aug = data_aug._load_aug_by_name()
val_datagen = DataGenerator(DATA_DIR, [VAL_ANNO_FILE1, VAL_ANNO_FILE2], aug, batch_size=24)

# ------------------ API --------------------- #

# Predict
@app.route("/predict")
def predict():
    return 'hello'


# ------------------ Helper functions --------------------- #

def test():
    # Get and preprocess image
    img_origin = val_datagen.load_image(TEST_IMAGE_PATH)
    preprocessors = [val_datagen.resize_img, val_datagen.mean_substraction]

    img_resize = cv2.resize(img_origin, (224, 224))[..., ::-1]
    img_preprocess = val_datagen.preprocessing(img_origin, preprocessors=preprocessors)

    img = np.expand_dims(img_preprocess, axis=0)

    # Predict
    prediction = model.predict(img)
    prediction = prediction[0]
    # print(prediction)

    # Mask
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

    # Return value
    cv2.imshow('', new_img)
    cv2.imwrite(OUTPUT_IMAGE_PATH, new_img)


# ------------------ Main --------------------- #

if __name__ == '__main__':
    # app.run(host=HOST, port=PORT, debug=DEBUG)
    test()
