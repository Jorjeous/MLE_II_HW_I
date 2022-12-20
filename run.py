import numpy as np
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
import os

main_dir = config['DEFAULT']['path']
test_dir = config['DATA']['test_dir']
data_dir = config['DATA']['path']
from keras.models import load_model
model = load_model(config["MODEL"]['path'])



path = os.path.join(data_dir,test_dir)
X_test = []
id_line = []
def create_test1_data(path):
    for p in os.listdir(path):
        id_line.append(p.split(".")[0])
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_test.append(new_img_array)
create_test1_data(path)
X_test = np.array(X_test).reshape(-1,80,80,1)
X_test = X_test/255

predictions = model.predict(X_test)

num = config['VAR']['img_num']
#predictions[int(num)]
#plt.imshow(X_test[num],cmap="gray")
print(num)
print(predictions[int(num)])
print(round(predictions[int(num)][0]))
