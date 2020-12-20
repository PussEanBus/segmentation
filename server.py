import os
import warnings

# ------------------ Flask --------------------- #

from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename


# ------------------ CV --------------------- #

from model import SiNet
import numpy as np
import keras.backend as K
import cv2
from data_generator.datagenerator import DataGenerator
from data_generator.dataaugentation import DataAugmentation
from unet import get_mobile_unet
from PIL import Image
import matplotlib.image as mpimg


# ------------------ General config --------------------- #

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------ Flask config --------------------- #

HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'outputs')
CSS_FOLDER = os.path.join(STATIC_FOLDER, 'css')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
CORS(app)

# ------------------ CV config --------------------- #

# Image config
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
DATA_DIR = os.path.join(BASE_DIR, 'Nukki')
VAL_ANNO_FILE1 = os.path.join(DATA_DIR, "baidu_V1", "val.txt")
VAL_ANNO_FILE2 = os.path.join(DATA_DIR, "baidu_V2", "val.txt")
N_CLASSES = 2

# Model
WEIGHT_FILE_PATH = os.path.join(BASE_DIR, 'weights', 'best_weights_4_all.h5')
IMAGE_DATA_FORMAT = K.image_data_format()
K.clear_session()
sinet = SiNet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL, N_CLASSES)
model = sinet.build_decoder()
model.load_weights(WEIGHT_FILE_PATH)

# unet
unet = get_mobile_unet(pretrained=True)
# data loader
data_aug = DataAugmentation()
aug = data_aug.load_aug_by_name()
val_datagen = DataGenerator(DATA_DIR, [VAL_ANNO_FILE1, VAL_ANNO_FILE2], aug, batch_size=24)


# ------------------ API --------------------- #

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    model_type = request.form.get('model_type')

    # TODO: get color

    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect('/')

    if file and allowed_file(file.filename) and model_type:
        # Save uploaded file
        new_name = f'{model_type}_{file.filename}'
        filename = secure_filename(new_name)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Predict
        if model_type == 'unet':
            output_file, output_file_path = predict_image_unet(filepath, filename)
        else:
            output_file, output_file_path = predict_image(filepath, filename)

        input_url = url_for('uploaded_file', filename=filename)
        output_url = url_for('generated_file', filename=output_file)
        # return redirect(output_url)
        return render_template('results.html', before=input_url, after=output_url)

    # flash('Invalid')
    # return redirect('/')
    return 'Invalid request', 400


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('index.html')


@app.route('/results', methods=['GET'])
def show_results():
    return render_template('results.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/outputs/<filename>')
def generated_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/css/<filename>')
def css_file(filename):
    return send_from_directory(CSS_FOLDER, filename)


# ------------------ Helper functions --------------------- #

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image_unet(file_path, file_name):
    print("unet prediction")

    im = Image.open(file_path)
    im = im.resize((128, 128), Image.ANTIALIAS)
    img = np.float32(np.array(im) / 255.0)

    # Reshape input and threshold output
    out = unet.predict(img[:, :, 0:3].reshape(1, 128, 128, 3))
    output_file_path = os.path.join(OUTPUT_FOLDER, file_name)
    mpimg.imsave(output_file_path, out[0] * img)
    return file_name, file_path


def predict_image(file_path, file_name):
    print("sinet prediction")

    # Get and preprocess image
    img_origin = val_datagen.load_image(file_path)
    preprocessors = [val_datagen.resize_img, val_datagen.mean_substraction]

    img_resize = cv2.resize(img_origin, (224, 224))[..., ::-1]
    img_preprocess = val_datagen.preprocessing(img_origin, preprocessors=preprocessors)

    img = np.expand_dims(img_preprocess, axis=0)
    # img = np.expand_dims(img_resize, axis=0)

    # Predict
    prediction = model.predict(img)
    prediction = prediction[0]

    # Mask
    mask = np.reshape(prediction, (IMG_HEIGHT, IMG_WIDTH, N_CLASSES))
    mask = np.argmax(mask, axis=-1)
    mask[mask > 0] = 255

    new_img = np.copy(img_resize)
    non_zeros_idx = np.where(mask == 0)
    new_img[..., 0][non_zeros_idx] = 0
    new_img[..., 1][non_zeros_idx] = 0
    new_img[..., 2][non_zeros_idx] = 0
    new_img[np.all(new_img == (0, 0, 0), axis=-1)] = (244, 118, 0)

    # mask = cv2.merge([mask, mask, mask])
    # img_resize = cv2.resize(img_resize, (512, 512))
    new_img = cv2.resize(new_img, (512, 512))

    # Return value
    output_file_path = os.path.join(OUTPUT_FOLDER, file_name)
    cv2.imwrite(output_file_path, new_img)
    return file_name, file_path


# ------------------ Main --------------------- #

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
