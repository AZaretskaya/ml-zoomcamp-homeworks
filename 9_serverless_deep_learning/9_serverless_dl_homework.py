#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', 'dino-vs-dragon-v2.tflite')

model = keras.models.load_model('dino_dragon_10_0.899.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('dino_dragon-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


interpreter = tflite.Interpreter(model_path='dino_dragon-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'dino',
    'dragon'
]

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
    image = download_image(url)
    image = prepare_image(image, target_size=(150, 150))
    x = np.array(image)
    X = np.array([x])
    image_preprocessed = np.float32(X * (1. / 255))


    interpreter.set_tensor(input_index, image_preprocessed)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_preds = preds[0].tolist()

    return dict(zip(classes, float_preds))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result