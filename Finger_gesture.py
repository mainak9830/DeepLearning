# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:39:32 2020

@author: Mainak
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io,transform
import os,glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,Activation
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers



train_img_list=glob.glob("/Users/Mainak/Desktop/spyder/fingers/train/*.png")

test_img_list=glob.glob("/Users/Mainak/Desktop/spyder/fingers/test/*.png")

print(len(train_img_list),len(test_img_list),sep='\n')

def import_data():
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    
    for img in train_img_list:
        img_read=io.imread(img)
        img_read=transform.resize(img_read,(128,128),mode='constant')
        X_train.append(img_read)
        y_train.append(img[-6:-4])
        
    for img in test_img_list:
        img_read=io.imread(img)
        img_read=transform.resize(img_read,(128,128),mode='constant')
        X_test.append(img_read)
        y_test.append(img[-6:-4])
    
    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

X_train,X_test,y_train,y_test=import_data()

X_train=np.expand_dims(X_train,axis=3)
X_test=np.expand_dims(X_test,axis=3)
label_to_int={
    '0R' : 0,
    '1R' : 1,
    '2R' : 2,
    '3R' : 3,
    '4R' : 4,
    '5R' : 5,
    '0L' : 6,
    '1L' : 7,
    '2L' : 8,
    '3L' : 9,
    '4L' : 10,
    '5L' : 11
}
def convert_label(y_train,y_test):
    temp=[]
    for label in y_train:
        temp.append(label_to_int[label])
    y_train=temp.copy()
    
    temp=[]
    
    for label in y_test:
        temp.append(label_to_int[label])
    y_test=temp.copy()
    
    return y_train,y_test

y_train,y_test=convert_label(y_train,y_test)

y_train=tf.keras.utils.to_categorical(y_train,num_classes=12)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=12)

#CNN network
weight_decay=1e-4
num_classes=12
model=Sequential()

model.add(Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(128,128,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 

model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation='linear'))
model.add(Activation('elu'))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(0.0003),metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train, batch_size=64, validation_data = (X_test,y_test), epochs = 5)

pred=model.predict(X_train[0:1])
predicted_class=np.argmax(pred)
print("predicted class--->",predicted_class,"actual class",y_train[0])