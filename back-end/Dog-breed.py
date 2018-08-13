# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:49:41 2018

@author: Govind Prabhu
"""

from flask import Flask, jsonify, request
import logging
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image                  
#from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from extract_bottleneck_features import *
import os
import cv2                
import matplotlib.pyplot as plt    
from keras.applications.resnet50 import ResNet50
import time


HOST = '0.0.0.0'
PORT = 5000
upload_folder = 'data/upload'


app = Flask(__name__)
app.config['UPLOAD_CONFIG'] = upload_folder
ResNet50_model = ResNet50(weights='imagenet')
dog_names = [item[20:-1] for item in sorted(glob("data/dog_images/train/*/"))]

train_xception = None
valid_xception = None
valid_xception = None
test_xception = None
test_targets = None 
xception_model = None


@app.route("/upload", methods=["post"])
def upload():
    print('request ', request, ' uploadFile ',request.files['uploadFile'])
    uploadFile = request.files['uploadFile']
    print('inside the upload method ', uploadFile.filename)
    fileName = uploadFile.filename
    uploadFile.save(os.path.join(app.config['UPLOAD_CONFIG'], fileName))
    print('File successfully saved')
    path = app.config['UPLOAD_CONFIG'] + '/'+ fileName
    return detectDogBreed(path)
    
@app.route('/train_from_scratch')
def train_data():
    train_files, train_targets = load_dataset('data/dog_images/train')
    valid_files, valid_targets = load_dataset('data/dog_images/valid')
    test_files, test_targets = load_dataset('data/dog_images/test')
    logging.debug('Data successfully loaded.. ')
    #print('input shape ', train_xception.shape[1:])
    load_xception_bottleneck()
    xception_model = Sequential()
    xception_model.add(GlobalAveragePooling2D(input_shape = train_xception.shape[1:]))
    xception_model.add(Dense(133, activation = 'softmax'))
    xception_model.summary()
    xception_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.RESNET50.hdf5', 
                               verbose=1, save_best_only=True)
    xception_model.fit(train_xception, train_targets,validation_data=(valid_xception, valid_targets),epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
    return jsonify({'Status':'Success_trained_data'})

def load_xception_bottleneck():
    bottleneck_features = np.load('data/bottleneck_features/DogXceptionData.npz')
    global train_xception, valid_xception, test_xception
    train_xception = bottleneck_features['train']
    valid_xception = bottleneck_features['valid']
    test_xception = bottleneck_features['test']
    

@app.route('/load_trained_weights')
def load_trained_weights():
    #bottleneck_features = np.load('data/bottleneck_features/DogXceptionData.npz')
    #train_xception = bottleneck_features['train']
    #valid_xception = bottleneck_features['valid']
    #test_xception = bottleneck_features['test']
    #test_files, test_targets = load_dataset('data/dog_images/test')
    #print(train_xception.shape[1:])
    global xception_model 
    xception_model = Sequential()
    xception_model.add(GlobalAveragePooling2D(input_shape = (7, 7, 2048)))
    xception_model.add(Dense(133, activation = 'softmax'))
    xception_model.load_weights('saved_models/weights.best.RESNET50.hdf5')
    #xception_prediction = [np.argmax(xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_xception]
    #test_accuracy = 100 * np.sum(np.array(xception_prediction) == np.argmax(test_targets, axis=1)) / len(xception_prediction)
    #print('Test accuracy: %.4f%%' % test_accuracy)
    return jsonify({'Status': 'Success_load_trained_weights'})


def path_to_tensor(img_path):
    print('Inside the image path ', img_path)
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets



def detectDogBreed(img_path):
    msg = ''
    if dog_detector(img_path):
        msg = 'Hey Dog! Let\'s see which breed you are'
    elif face_detector(img_path):
        msg = 'Hey Human! Let\'s see which type of dog breed you resembles with..'
    else:
        msg = ' :( Neither a dog nor a human. Try with an image of either of them'
        return 
    plt.imshow(image.load_img(img_path))
    plt.show()
    res = resNet50_predict_breed(img_path)
    print('result from res ', res)
    return jsonify({'message':msg, 'result' : res})
    
def face_detector(img_path):
    print('Image path fac detector '+ img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def resNet50_predict_breed(img_path):
    start_time = time.time()
    global xception_model
    if xception_model is None:
        print('Xception model is none. Loading.. ')
        load_trained_weights()
        print('Xception model is loaded ')
        
    #print('Summery ', xception_model.summary())
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    print('Got bottleneck features')
    # obtain predicted vector
    predicted_vector = xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    print("%s second time "%(time.time() - start_time))
    return dog_names[np.argmax(predicted_vector)]


if __name__ == "__main__":    
    #print('Running the app')
    app.run()
