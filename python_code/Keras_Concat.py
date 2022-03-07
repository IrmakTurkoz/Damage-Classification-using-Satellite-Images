# We use this code for the Concat VGG16 model

import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D

diseaster = "wind/"

DATADIR1 = "post/" + diseaster
DATADIR2 = "pre/" + diseaster

CATEGORIES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
IMG_SIZE = 224

training_data1 = []
training_data2 = []

def create_training_data1():
    for category in CATEGORIES:
        path = os.path.join(DATADIR1,category)
        class_num = CATEGORIES.index(category)
        onehot = [0,0,0,0]
        onehot[class_num] = 1
        ccount = 1
        for img in tqdm(os.listdir(path)):
          try:
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR) #_COLOR
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data1.append([new_array, onehot])
            ccount = ccount + 1
            if(ccount > 2250):
                break
          except:
            print("An exception occurred")

def create_training_data2():
    for category in CATEGORIES:
        path = os.path.join(DATADIR2,category)
        class_num = CATEGORIES.index(category)
        onehot = [0,0,0,0]
        onehot[class_num] = 1
        ccount = 1
        for img in tqdm(os.listdir(path)):
          try:
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR) #_COLOR
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data2.append([new_array, onehot])
            ccount = ccount + 1
            if(ccount > 2250):
                break
          except:
            print("An exception occurred")

create_training_data1()
create_training_data2()

print(len(training_data1))
print(len(training_data2))

import random
random.seed(7) # to get the same shuffle for pre and post data

random.shuffle(training_data1)
random.shuffle(training_data2)

X1 = []
X2 = []
y = []

for features,label in training_data1:
    X1.append(features)
    y.append(label)

del training_data1
X1 = np.array(X1).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y).reshape(-1, 4)


for features,label in training_data2:
    X2.append(features)
del training_data2
X2 = np.array(X2).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Normalize X1 and X2
X1 = X1/255.0
X2 = X2/255.0

# Finetune Merged VGG16 Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model

vgg_conv_C = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_conv_D = VGG16(weights='imagenet', include_top=False, input_shape= (224, 224, 3))

for layer in vgg_conv_D.layers[:-1]:
    layer.trainable = False
for layer in vgg_conv_D.layers[:]:
    layer._name = 'One_' + layer.name
for layer in vgg_conv_C.layers[:-1]:
    layer.trainable = False
for layer in vgg_conv_C.layers[:]:
    layer._name = 'Two_' + layer.name
mergedModel = concatenate([vgg_conv_C.output,vgg_conv_D.output])
mergedModel = layers.Flatten()(mergedModel)
mergedModel = layers.Dense(128)(mergedModel)
mergedModel = layers.Activation('relu')(mergedModel)
mergedModel = layers.Dense(64,activation = 'relu')(mergedModel)
mergedModel = layers.Dense(4,activation = 'softmax')(mergedModel)
fused_model = Model([vgg_conv_C.input, vgg_conv_D.input], mergedModel)  

# fused_model.summary()

fused_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from sklearn.utils import class_weight
y_integers = np.argmax(y, axis=1)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
class_weights = dict(enumerate(class_weights))

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
mcp_save = ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

fused_model.fit([X1,X2], y, batch_size=32, epochs=100, validation_split=0.2, shuffle=True, class_weight=class_weights, callbacks=[mcp_save])