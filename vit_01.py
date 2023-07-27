#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Was vit_anom.py
"""
##===================== MEMORY FRAGMENTED =========================##
#import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
##=================================================================##
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import os, time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD DATA SHAPED DATA - All images                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#class_types = ['folding_marks', 'grain_off', 'growth_marks', 'loose_grains', 'non_defective', 'pinhole']
class_types = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']

#img_data = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_07b_data.npy')   # Load the data
#labels   = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_07b_label.npy')  # Load the label
img_data = np.load('/home/antz/Downloads/MEng/datasets/kaggle_09_data.npy')  # Load the data
labels  = np.load('/home/antz/Downloads/MEng/datasets/kaggle_09_label.npy')  # Load the labels

print("[INFO] TRAINING SET SHAPE: {}".format(img_data.shape))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 6 CLASSES & DATA PREPARATION                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 6                                                 # define the number of classes
#input_shape = (32, 32, 3)                                      # Use Kaggle_02 - Mem_MAX
input_shape = (224, 224, 3)                                     # Use Kaggle_07
#num_of_samples = img_data.shape[0]                              # Get the number of samples

# One-Hot Encoding of labels #
Y = np_utils.to_categorical(labels,num_classes)
# Shuffle data              #
x,y = shuffle(img_data,Y,random_state=2)
# Split data - Train/Test   #
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

print("[INFO] PRE-PROCESSING COMPLETE --")

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Configure the hyperparameters
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
learning_rate   = 0.001
weight_decay    = 0.0001
batch_size      = 96                # reduced from 256
num_epochs      = 100               # same
image_size      = 72                # We'll resize input images to this size 
patch_size      = 6                 # Size of the patches to be extract from the input images
num_patches     = (image_size // patch_size) ** 2
projection_dim  = 64
num_heads       = 4
transformer_units = [projection_dim * 2, projection_dim, ] # Size of the transformer layers
transformer_layers = 8
#mlp_head_units = [2048, 1024]      # Size of the dense layers of the final classifier
mlp_head_units  = [512, 256]        # Change mlp_head_units

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Use data augmentation
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization
data_augmentation.layers[0].adapt(x_train)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Implement multilayer perceptron (MLP)
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Implement patch creation as a layer
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Implement the patch encoding layer:
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Locality Self Attention: MultiHeadAttention
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Implemented in "Model Build ##

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Build the ViT model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # Add final dense layer with softmax activation.
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## ~~ MODEL EVALUATION FUNCTION ~~ ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
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

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Compile, train, and evaluate the mode
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
def run_experiment(model):
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', f1_m, precision_m, recall_m, tf.keras.metrics.AUC(name='auc')])

    checkpoint_filepath = "tmp"    #"/tmp/checkpoint"
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    earlystop = EarlyStopping(
        monitor='val_accuracy', 
        patience=12, 
        mode='max', 
        verbose=1)

    ##---------------------------------------------##
    ##                 ~ MODEL FIT ~               ##
    ##---------------------------------------------##
    t=time.time()
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,                          # 64
        epochs=num_epochs,                              # 100
        validation_split=0.2,                           # was 0.1
        callbacks=[checkpoint_callback],                # put back: checkpoint_callback, 
    )
    ##----------------------------------##
    ##      EVALUATE THE MODEL          ##
    ##----------------------------------##
    #loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    #loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    loss, accuracy, f1_score, precision, recall, auc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    
    # View Results
    print('[INFO] Training time: %s' % (t - time.time()))
    print(f"[INFO] Test accuracy: {round(accuracy * 100, 2)}%")
    print("[INFO] loss={:.4f}".format(loss))
    print("[INFO] precision={:.4f}%".format(precision))
    print("[INFO] recall={:.4f}%".format(recall))
    print("[INFO] f1_score={:.4f}%".format(f1_score))
    print("[INFO] AUC={:.4f}%".format(auc))

    ##----------------------------------##
    ##      RETURN METRICS & MODEL      ##
    ##----------------------------------##
    return history, model

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Visualise the Training Metrics                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def plot_metrics(history):
    #epoch_stop = len(history.history['loss'])                    # returns epoch count
    epoch_stop = len(history.history['val_accuracy'])
    print("[INFO] Epoch Stopped: {}".format(epoch_stop))

    # visualizing losses and accuracy
    train_loss = history.history['loss']
    train_acc  = history.history['accuracy']
    val_loss   = history.history['val_loss']
    val_acc    = history.history['val_accuracy']
    train_prec = history.history['precision_m']
    val_prec   = history.history['val_precision_m']
    train_recall=history.history['recall_m']
    val_recall = history.history['val_recall_m']
    xc=range(epoch_stop)                        # number of epochs

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_accuracy vs val_accuracy')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available                      # use bmh, classic, ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig('ViT_01_Acc_96_100.png')

    """
    fig, ax2 = plt.subplots(1,figsize=(7,5))
    ax2.set_title('train_loss vs val_loss')
    ax2.set_xlabel('num of Epochs')
    ax2.set_ylabel('loss')
    ax2.plot(xc,train_loss)
    ax2.plot(xc,val_loss)
    ax2.legend(['train','val'],loc=1)
    ax2.grid(color='0.75', linestyle='-', linewidth=0.5)
    ax2.set_ylim([0, 1])                                    # Limit graph to <1
    plt.savefig('cust03_loss_k3_100_120.png')
    """
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'],loc=1)
    #print plt.style.available                      # use bmh, classic, ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig('ViT_01_Loss_96_100.png')

    plt.figure(3,figsize=(7,5))
    plt.plot(xc,train_prec)
    plt.plot(xc,val_prec)
    plt.xlabel('num of Epochs')
    plt.ylabel('precision')
    plt.title('train_precision vs val_precision')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available                      # use bmh, classic, ggplot for big pictures
    plt.style.use(['classic'])
    #plt.savefig('Prec_incept_32_12.png')
    plt.savefig('ViT_01_Prec_96_100.png')

    plt.figure(4,figsize=(7,5))
    plt.plot(xc,train_recall)
    plt.plot(xc,val_recall)
    plt.xlabel('num of Epochs')
    plt.ylabel('recall')
    plt.title('train_recall vs val_recall')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available                      # use bmh, classic, ggplot for big pictures
    plt.style.use(['classic'])
    #plt.savefig('Recall_incept_32_12.png')
    plt.savefig('ViT_01_Rec_96_100.png')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Plots conf. matrix and classification report
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def conf_matrix(model, X_test, Y_test):

    predictions = model.predict(X_test)
    X_pred = np.argmax(np.round(predictions), axis=1)
    print(f"first: {X_pred[1]}")
    Y_pred = np.argmax(Y_test, axis=1)
    print(f"first: {Y_pred[1]}")
    
    cm = confusion_matrix(Y_pred, X_pred)
    
    print("Classification Report:\n")
    
    cr = classification_report(Y_pred,
                                np.argmax(np.round(predictions), axis=1), 
                                target_names=[class_types[i] for i in range(len(class_types))])
    print(cr)   # change to save to file?
    
    plt.figure(figsize=(12,12))
    sns_hmp = sns.heatmap(cm, annot=True, xticklabels = [class_types[i] for i in range(len(class_types))], 
                yticklabels = [class_types[i] for i in range(len(class_types))], fmt="d")
    fig = sns_hmp.get_figure()
    plt.savefig('ViT_01_matrix.png')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Make sing Prediction
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

##===============================================================##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Train and Evaluate
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
vit_classifier = create_vit_classifier()
history, model = run_experiment(vit_classifier)

##===============================================================##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Plot Performance Metrics & Prediction
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
plot_metrics(history)
conf_matrix(model, x_test, y_test)             # pred_class = vit_classifier.predict(x_test)


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
