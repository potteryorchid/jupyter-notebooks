#!/usr/bin/env python
# coding: utf-8

# ### 数据-模型-服务流水线

# <img width=60% height=60% src="imgs/18/01.png" alt="imgs/18/01.png" title="图1" />

# ### 部署验证码识别服务
# #### 另存为app.py，启动flask来加载app.py文件（存放路径 "./code"）

# In[4]:


import base64

import numpy as np
import tensorflow as tf

from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER # captcha charset number
CAPTCHA_LEN = 4 # captcha length
CAPTCHA_HEIGHT = 60 # captcha height
CAPTCHA_WIDTH = 160 # captcha width

MODEL_FILE = '../../models/16/captcha_rmsprop_binary_crossentropy_bs_100_epochs_30.h5'

def rgb2gray(img):
    # y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text

app = Flask(__name__)

@app.route('/ping', methods=['GET', 'POST'])
def hello_word():
    return 'pong'

@app.route('/predict', methods=['post'])
def predict():
    response = {'success': False, 'prediction': '', 'debug': 'error'}
    received_image = False
    if request.method == 'POST':
        if request.files.get('image'):
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'
        elif request.get_json():
            encoded_image = request.get_json()['image']
            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(1, 60, 160, 1).astype('float32') / 255
            with graph.as_default():
                pred = model.predict(image)
            response['prediction'] = response['prediction'] + vec2text(pred)
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)

model = load_model(MODEL_FILE)
graph = tf.get_default_graph()


# In[2]:




