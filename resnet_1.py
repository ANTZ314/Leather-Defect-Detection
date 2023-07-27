# -*- coding: utf-8 -*-
"""
======================================================
Description:
	Modifies ResNet50 from 1000 output classes to 6 output classes
	Dataset: kaggle_07: dim[3595, 224, 224, 3] + 5 test images

STEPS:
	[1] Load and re-shape FULL datset images (6 folders) - Store as NumPy file
	[2] Image pre-processing - 6 classes, labels, normalise, 1-hot encode, shuffle, split
	[3] Create custom model from ResNet-50 - 1000 Classes to 6 Classes
	[4] Train the Cusom Model - Freeze all layers but last
		-> Make all layers untrainable ()
		-> Except Last-Layer is trainable
		-> Compile new Model
		-> Fitting & Validating
	[5] Testing & Evaluating the Custom Model - Fitting
	[6] TEST MODEL STORAGE AND RETREIVAL?
	[7] Single Image Prediction:
		-> Convert single image data to correct format 
		-> Modified Model Prediction on single test image
======================================================
"""
## Import and Shape Dataset ##
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
## Modify ResNet-50 Model ##
from keras.applications.resnet import ResNet50
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten
## Analysis ##
import time, os
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow import keras


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD DATA SHAPED DATA - All images                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
no_batch = 100  # 100
no_epoch = 120  # 500

img_data = np.load('/home/antz/Downloads/MEng/datasets/kaggle_09_data.npy')  # Load the data
labels   = np.load('/home/antz/Downloads/MEng/datasets/kaggle_09_label.npy') # Load the label

print("[INFO] TRAINING SET SHAPE: {}".format(img_data.shape))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 6 CLASSES & DATA PREPARATION                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 6  												# define the number of classes
num_of_samples = img_data.shape[0]								# Get the number of samples

names = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']

# One-Hot Encoding of labels #
Y=np_utils.to_categorical(labels,num_classes)
# Shuffle data              #
x,y = shuffle(img_data,Y,random_state=2)
# Split data - Train/Test   #
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print("[INFO] PRE-PROCESSING SUCCESSFULL --")


#=================================================================#
# Create a Custom ResNet Model - 1000 Classes to 6 Classes        #
#=================================================================#
print("-- MODIFYING RESNET-50 MODEL --")

## Training the classifier alone
image_input = Input(shape=(224, 224, 3))

## 'top=True'= Include the last layer
model = ResNet50(input_tensor = image_input, include_top = True, weights = 'imagenet')

## Extract last pooling layer (before 1000 class Dense layer)
last_layer = model.get_layer('avg_pool').output
x = Flatten(name = 'flatten')(last_layer)

## Create a new last-layer
out = Dense(num_classes, activation = 'softmax', name = 'output_layer')(x)

## Create Custom Model using new "last-layer" with 6 Classes
model = Model(inputs = image_input, outputs = out)
#model.summary()							# View the new model (6 Classes)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Train the Cusom Model - FREEZE ALL LAYERS EXCEPT LAST LAYER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Make all layers untrainable ()
for layer in model.layers[:-1]:
	layer.trainable = False

# Except Last-Layer is trainable
model.layers[-1].trainable

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


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

t=time.time()

## train the model (default batch size 32) ##
history = model.fit(X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size      = no_batch,
    epochs          = no_epoch,
    callbacks       = callback)

#print(history.history.keys())

print('[INFO] Training time: %s' % (t - time.time()))

## evaluate the model ##
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, batch_size=no_batch, verbose=0)

# View Results
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("[INFO] precision={:.4f}%".format(precision * 100))
print("[INFO] recall={:.4f}%".format(recall * 100))
print("[INFO] f1_score={:.4f}%".format(f1_score * 100))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Get Epoch Stopped At:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
epoch_stop = len(history.history['loss'])                    # returns epoch count
print("[INFO] Epoch Stopped: {}".format(epoch_stop))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Visualise the Training Metrics                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# visualizing losses and accuracy
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
plt.savefig('resnet1_loss_100_120_k3.png')

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
plt.savefig('resnet1_acc_100_120_k3.png')

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
plt.savefig('resnet1_prec_100_120_k3.png')

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
plt.savefig('resnet1_rec_100_120_k3.png')

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
