#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGPT generated EfficientNetB0 on 6 class
Requirement:
pip install efficientnet

Using Kaggle_06 - Already split int 480:120 each
"""
## Import the necessary libraries and packages:

##===================== MEMORY FRAGMENTED =========================##
import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
##=================================================================##
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from efficientnet.tfkeras import EfficientNetB0

import time, os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
#from keras.preprocessing import image
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Set the image size, Batch-Size, Epochs and Dataset:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
IMG_SIZE    = 224
BATCH_SIZE  = 32
NO_EPOCHS   = 100

#train    = '/home/antz/Documents/0_models/dataset/kaggle_06/train'          # 480 each
#validate = '/home/antz/Documents/0_models/dataset/kaggle_06/validate'       # 120 each
train    = '/home/antz/Downloads/kaggle_06/train'                               # 475 each
validate = '/home/antz/Downloads/kaggle_06/validate'                            # 115 each

names = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Define the data generator for loading the dataset:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=20, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   vertical_flip=True, 
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(directory=train, 
                                                    target_size=(IMG_SIZE, IMG_SIZE), 
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='categorical', 
                                                    subset='training')

valid_generator = train_datagen.flow_from_directory(directory=train, 
                                                    target_size=(IMG_SIZE, IMG_SIZE), 
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='categorical', 
                                                    subset='validation')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Define the model architecture:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## ~~ MODEL EVALUATION FUNCTION ~~ ##
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

## Compile the model:
#model.compile(optimizer=Adam(learning_rate=0.0001), 
#              loss='categorical_crossentropy', 
#              metrics=['accuracy', f1_m, precision_m, recall_m])

model.compile(optimizer="adam", #Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', f1_m, precision_m, recall_m, tf.keras.metrics.AUC(name='auc')])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Train the model:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
checkpoint = ModelCheckpoint('/home/antz/Downloads/MEng/datasets/efficientnet-b0_weights.h5', 
                              monitor='val_accuracy', 
                              save_best_only=True, 
                              save_weights_only=True, 
                              mode='max', 
                              verbose=1)

earlystop = EarlyStopping(monitor='val_accuracy', 
                          patience=10, 
                          mode='max', 
                          verbose=1)

t=time.time()

hist = model.fit(train_generator, 
                    batch_size      = BATCH_SIZE,
                    epochs          = NO_EPOCHS, 
                    verbose         = 1, 
                    validation_data = valid_generator, 
                    callbacks       = [checkpoint, earlystop])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## ~~ MODEL EVALUATION FUNCTION ~~ ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(directory=validate, 
                                                  target_size=(IMG_SIZE, IMG_SIZE), 
                                                  batch_size=BATCH_SIZE, 
                                                  class_mode='categorical')

## evaluate the model ##
loss, accuracy, f1_score, precision, recall, auc = model.evaluate(test_generator, batch_size=BATCH_SIZE, verbose=0)

# View Results
print('[INFO] Training time: %s' % (t - time.time()))
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("[INFO] precision={:.4f}%".format(precision))
print("[INFO] recall={:.4f}%".format(recall))
print("[INFO] f1_score={:.4f}%".format(f1_score))
print("[INFO] AUC={:.4f}%".format(auc))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Get Epoch Stopped At:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
epoch_stop = len(hist.history['loss'])                    # returns epoch count
print("[INFO] Epoch Stopped: {}".format(epoch_stop))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Visualise the Training Metrics                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# visualizing losses and accuracy
train_loss=hist.history['loss']
train_acc=hist.history['accuracy']
val_loss=hist.history['val_loss']
val_acc=hist.history['val_accuracy']
train_prec=hist.history['precision_m']
val_prec=hist.history['val_precision_m']
train_recall=hist.history['recall_m']
val_recall=hist.history['val_recall_m']

precision_m
xc=range(epoch_stop)                        # number of epochs

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'],loc=1)
#print plt.style.available                  # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('eff_loss_32_100.png')

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
plt.savefig('eff_acc_32_100.png')

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
plt.savefig('eff_prec_32_100.png')

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
plt.savefig('eff_rec_32_100.png')

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
