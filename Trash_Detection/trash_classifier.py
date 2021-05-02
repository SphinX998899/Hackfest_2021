import keras
from pypylon import pylon
from keras.models import Model, load_model
from keras.applications import mobilenet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os

def pp_image(img):
    img = image.load_img('pic.png', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return np.asarray(x)

prediction_list=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model=load_model('model1.h5', custom_objects={'relu6': keras.layers.ReLU(6.)})

camera = cv2.VideoCapture(0);

while True:
    time.sleep(0.005)
    ret, img = camera.read()
    pred_img=pp_image(img)
    yo=model.predict(pred_img)
    pred=prediction_list[np.argmax(yo)]
    cv2.putText(img, pred, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,0), 5, False)
    cv2.imshow('image',img);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()