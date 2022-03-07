# Mount Google Drive to get data, requires authorization
from google.colab import drive
drive.mount('/content/drive')

# Load and copy zip file
# This should be replaced by the sample set of interest
zip_path = "/content/drive/My Drive/dataset/Post/earthquake.zip"
!cp "{zip_path}" .

# Unzip and remove the zip file
!unzip -q earthquake.zip
!rm earthquake.zip

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

CATEGORIES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
DATADIR = "earthquake"
training_data = []

IMG_SIZE = 224 # Decided by VGG16

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    class_num = CATEGORIES.index(category)
    onehot = [0,0,0,0]
    onehot[class_num] = 1
    ccount = 1	
    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR) # Read in color for VGG16
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, onehot])
        ccount = ccount + 1
        if(ccount > 1500): # This is used for limiting the input size per class
          break

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

del training_data # save memory

# Numpy arrays for the input
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y).reshape(-1, 4)

X = X/255.0 # Normalize

# Split dataset as training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle= True)

# For data augmentation
datagen = ImageDataGenerator(
  rotation_range=45,
  zoom_range=0.2,
  width_shift_range=0.05,
  height_shift_range=0.03,
  shear_range=0.05,
  fill_mode="nearest",
  horizontal_flip=True,
  vertical_flip=True)

training_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
datagen.fit(X_train)

# Import VGG16 for finetuning
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
model.add(Flatten())
 
# Add new layers
model.add(layers.Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(layers.Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(layers.Dense(4, activation='softmax'))
 
opt = optimizers.Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

mcp_save = ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

from sklearn.utils import class_weight
y_integers = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
class_weights = dict(enumerate(class_weights))

history = model.fit_generator(training_generator,steps_per_epoch=(len(X_train))//32, epochs=175, 
	validation_data=(X_valid,y_valid), callbacks=[mcp_save])#, class_weight=class_weights)

# summarize history for accuracy
plt.figure(figsize=(6,5))
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png', dpi=300)

print(max(model.history.history['val_acc']))

# Load the best model back for report
from tensorflow.keras.models import load_model
model = load_model('model.hdf5')
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_valid.argmax(axis=1), model.predict(X_valid).argmax(axis=1)))
print(classification_report(y_valid.argmax(axis=1), model.predict(X_valid).argmax(axis=1)))