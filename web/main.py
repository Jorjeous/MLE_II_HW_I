from flask import Flask, request, render_template
import os
#import cv2
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
#import configparser


#X_test = []
#id_line = []
#def create_test1_data(path):
#    for p in os.listdir(path):
#        id_line.append(p.split(".")[0])
#        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
#        new_img_array = cv2.resize(img_array, dsize=(80, 80))
#        X_test.append(new_img_array)
#create_test1_data('../data/test1')
#X_test = np.array(X_test).reshape(-1,80,80,1)
#X_test = X_test/255
#from keras.models import load_model
#model = load_model('../data/model_sv.h5')

#predictions = model.predict(X_test)

with open('../results.txt', 'r') as f:
    imgnums = f.read()



app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    num = int(text)
    ans = 'This is...   '
    if imgnums[num] == '1':
        animal = 'dog'
    else:
        animal = 'cat'

    return ans+animal
