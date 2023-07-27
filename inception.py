#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Modifies InceptionV3 from 1000 output classes to 6 output classes
    Dataset: class_6: dim[224, 224, 3], Train(480), Test(120)

STEPS:
    [1] Load Pre-Trained Model, re-shape test input image & make un-modified prediction
    [2] Load datasets and re-shape all images to match formatting
    [3] Create 4 classes and 1-hot encode
    [4] Modify Inception Model - from 1000 to 4 output classes
    [5] RESULTS:
"""

##===================== MEMORY FRAGMENTED =========================##
import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
##=================================================================##
#import tensorflow as tf
from tensorflow import keras
import numpy as np
## Note: If model is not there it will auto-download
from keras.applications.inception_v3 import InceptionV3
## TO CREATE A CUSTOM MODEL ##
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.layers import Input
## Dataset manipulation ##
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
## Prediction ##
import time, os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import decode_predictions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Set Batch-Size, Epochs and Load Dataset                       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
no_batch = 32   #32
no_epoch = 100  #120

#img_data = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_07b_data.npy')   # Load the data
#labels   = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_07b_label.npy')  # Load the label
img_data = np.load('/home/antz/Downloads/MEng/datasets/kaggle_09_data.npy')  # Load the data
labels   = np.load('/home/antz/Downloads/MEng/datasets/kaggle_09_label.npy') # Load the label

print("[INFO] TRAINING SET SHAPE: {}".format(img_data.shape))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 6 CLASSES & DATA PREPARATION                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 6                                                 # define the number of classes
num_of_samples = img_data.shape[0]                              # Get the number of samples

names = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']

# One-Hot Encoding of labels #
Y=np_utils.to_categorical(labels,num_classes)
# Shuffle data              #
x,y = shuffle(img_data,Y,random_state=2)
# Split data - Train/Test   #
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print("[INFO] PRE-PROCESSING SUCCESSFULL --")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## MODIFY EXISTING INCEPTION_V3 MODEL - UNTESTED??
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

model = InceptionV3()

print("-- MODIFYING INCEPTION-V3 MODEL --")

## Custom Model - Define new input layer dimmensions
image_input = Input(shape=(224,224,3))      

## Custom Model - Define input tensors
model = InceptionV3(input_tensor=(image_input), include_top='True', weights='imagenet')

## Get output of the max pooling layer (last before resnet output layer)
last_layer = model.get_layer('avg_pool').output         # Has all ResNet layers before it
x = Flatten(name='flatten')(last_layer)                 # Added new flatten layer "x"

## Add new Dense layer with 4 Classes as Output
out = Dense(num_classes, activation=('softmax'), name='output_layer')(x)

## Custom Model - Create the Model
model = Model(inputs=image_input, outputs=out)

#model.summary()                           # Dimensions = 4 classes: Dense (None, 4)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Train the Custom Model - FREEZE ALL LAYERS EXCEPT LAST LAYER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#----------------------# From class_6_1.py #----------------------#
"""
# Make all layers untrainable ()
for layer in model.layers[:-1]:
    layer.trainable = False

# Except Last-Layer is trainable
model.layers[-1].trainable
"""
#----------------------# From class_6_1.py #----------------------#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## ~~ MODEL EVALUATION FUNCTIONS ~~ ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Compile and Fit the Model ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy', f1_m, precision_m, recall_m, tf.keras.metrics.AUC(name='auc')])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

t=time.time()

## train the model (default batch size 32) ##
history = model.fit(X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size      = no_batch,
    epochs          = no_epoch,
    callbacks       = callback)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Evaluate the model ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
loss, accuracy, f1_score, precision, recall, auc = model.evaluate(X_test, y_test, batch_size=no_batch, verbose=0)

# View Results
print('[INFO] Training time: %s' % (t - time.time()))
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("[INFO] precision={:.4f}%".format(precision))
print("[INFO] recall={:.4f}%".format(recall))
print("[INFO] f1_score={:.4f}%".format(f1_score))
print("[INFO] AUC={:.4f}%".format(auc))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Visualise the Training Metrics                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
epoch_stop = len(history.history['loss'])                    # returns epoch count
print("[INFO] Epoch Stopped: {}".format(epoch_stop))

## visualizing losses and accuracy ##
train_loss=history.history['loss']
train_acc=history.history['accuracy']
val_loss=history.history['val_loss']
val_acc=history.history['val_accuracy']
train_prec=history.history['precision_m']
val_prec=history.history['val_precision_m']
train_recall=history.history['recall_m']
val_recall=history.history['val_recall_m']
xc=range(epoch_stop)                        # number of epochs

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available                  # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
#plt.savefig('Loss_incept_32_12.png')
plt.savefig('incept_Loss_32_100.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_accuracy vs val_accuracy')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available                  # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
#plt.savefig('Acc_incept_32_12.png')
plt.savefig('incept_Acc_32_100.png')

plt.figure(3,figsize=(7,5))
plt.plot(xc,train_prec)
plt.plot(xc,val_prec)
plt.xlabel('num of Epochs')
plt.ylabel('precision')
plt.title('train_precision vs val_precision')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available                  # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
#plt.savefig('Prec_incept_32_12.png')
plt.savefig('incept_Prec_32_100.png')

plt.figure(4,figsize=(7,5))
plt.plot(xc,train_recall)
plt.plot(xc,val_recall)
plt.xlabel('num of Epochs')
plt.ylabel('recall')
plt.title('train_recall vs val_recall')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available                  # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
#plt.savefig('Recall_incept_32_12.png')
plt.savefig('incept_Rec_32_100.png')

print("[INFO] TRAINING METRICS PLOTTED")


##===============================================================##
## 						PREDICTION TEST 						 ##
##===============================================================##
def make_prediction(model, img_path, image_name):
    # Check Order:
    names = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']
    
    ## Convert single image data to correct format ##
    img=image.load_img(img_path, target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)

    feature= model.predict(x)
    print(feature)                                          # 
    prediction = np.argmax(feature)
    print(f"[INFO] Input Image: {image_name}")
    print(f"[INFO] Prediction:  {names[prediction]}")       # 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Prediction & Anomaly:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_path_0='/home/antz/Downloads/MEng/datasets/kaggle_10/test/folding_marks'
img_path_1='/home/antz/Downloads/MEng/datasets/kaggle_10/test/growth_marks'
img_path_2='/home/antz/Downloads/MEng/datasets/kaggle_10/test/pinhole'
img_path_3='/home/antz/Downloads/MEng/datasets/kaggle_10/test/grain_off'
img_path_4='/home/antz/Downloads/MEng/datasets/kaggle_10/test/non_defective'
img_path_5='/home/antz/Downloads/MEng/datasets/kaggle_10/test/loose_grains'


#~~~~~~~~~~~~~~~~ folding_marks ~~~~~~~~~~~~~~~~~~~~~#
arr = os.listdir(img_path_0)

for x in range(len(arr)):
    pred_img = img_path_0 + "/" + arr[x]
    #print(pred_img)
    make_prediction(model, pred_img, arr[x])
print("\n")

#~~~~~~~~~~~~~~~~ growth_marks ~~~~~~~~~~~~~~~~~~~~~#
arr = os.listdir(img_path_1)

for x in range(len(arr)):
    pred_img = img_path_1 + "/" + arr[x]
    make_prediction(model, pred_img, arr[x])
print("\n")

#~~~~~~~~~~~~~~~~ pinhole ~~~~~~~~~~~~~~~~~~~~~#
arr = os.listdir(img_path_2)

for x in range(len(arr)):
    pred_img = img_path_2 + "/" + arr[x]
    make_prediction(model, pred_img, arr[x])
print("\n")

#~~~~~~~~~~~~~~~~ grain_off ~~~~~~~~~~~~~~~~~~~~~#
arr = os.listdir(img_path_3)

for x in range(len(arr)):
    pred_img = img_path_3 + "/" + arr[x]
    make_prediction(model, pred_img, arr[x])
print("\n")

#~~~~~~~~~~~~~~~~ non_defective ~~~~~~~~~~~~~~~~~~~~~#
arr = os.listdir(img_path_4)

for x in range(len(arr)):
    pred_img = img_path_4 + "/" + arr[x]
    make_prediction(model, pred_img, arr[x])
print("\n")

#~~~~~~~~~~~~~~~~ loose_grains ~~~~~~~~~~~~~~~~~~~~~#
arr = os.listdir(img_path_5)

for x in range(len(arr)):
    pred_img = img_path_5 + "/" + arr[x]
    make_prediction(model, pred_img, arr[x])
