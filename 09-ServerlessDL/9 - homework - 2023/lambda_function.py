#!/usr/bin/env python
# coding: utf-8

# jupyter nbconvert --to script 'my-notebook.ipynb'

import tflite_runtime.interpreter as tflite
#import tensorflow.lite as tflite
import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

#interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')


interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

#url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    target_size=(150,150)
    img = prepare_image(img, target_size)

    x = np.array(img, dtype='float32') / 255.0
    # Turning this image into a batch of one image
    X = np.array([x])

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0][0].tolist()

    #return preds[0][0]
    return float_predictions

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

# Testing
# use ipython
# import lambda_function
# event = {'url': 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'}
# lambda_function.lambda_handler(event, None)