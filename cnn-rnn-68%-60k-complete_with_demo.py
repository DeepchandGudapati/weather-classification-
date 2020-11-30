#!/usr/bin/env python
# coding: utf-8

# In[33]:


from keras.applications.xception import Xception
from keras.applications import ResNet50V2
from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Lambda, LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.layers import merge
import time
import numpy as np

np.random.seed(1337)


# In[35]:


import keras
import keras.utils
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# In[3]:


import tensorflow as tf
import sys
import os


# In[4]:


train_datagen = ImageDataGenerator(
    validation_split=0.3)


# In[5]:


dirname = 'D:/200X200'
dir_chess_folders = os.listdir(dirname)
dir_chess_paths = [os.path.join(dirname, path) for path in dir_chess_folders]
dir_chess_paths


# In[6]:


batch_size_phase_one = 32
batch_size_phase_two = 16
nb_val_samples = 100

nb_epochs = 5

img_width = 200
img_height = 200


# In[7]:


def get_training_generator_test(batch_size=128):
    train_generator_test = train_datagen.flow_from_directory(
    dirname,
    target_size=(img_width, img_height),
    class_mode='categorical',
    subset='training',
    batch_size=batch_size)

    val_generator_test = train_datagen.flow_from_directory(
    dirname,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,)
    return train_generator_test, val_generator_test


# In[8]:


# Loading dataset
print("Loading the dataset with batch size of {}...".format(batch_size_phase_one))
train_generator_test, val_generator_test = get_training_generator_test(batch_size_phase_one)
print("Dataset loaded")

print("Building model...")
input_tensor = Input(shape=(img_width, img_height, 3))


# In[9]:


# Creating CNN
cnn_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = cnn_model.output
cnn_bottleneck = GlobalAveragePooling2D()(x)


# In[10]:


# Make CNN layers not trainable
for layer in cnn_model.layers:
    layer.trainable = False


# In[11]:


# Creating RNN
x=input_tensor
x = Reshape((30, 4000))(x)  # 23 timesteps, input dim of each timestep 3887
x = LSTM(2048, return_sequences=True)(x)
rnn_output = LSTM(2048)(x)


# In[12]:


# Merging both cnn bottleneck and rnn's output wise element wise multiplication
x = keras.layers.concatenate([cnn_bottleneck, rnn_output])
predictions = Dense(6, activation='softmax')(x)

model = Model(input=input_tensor, output=predictions)

print("Model built")


# In[13]:


from keras.callbacks import ReduceLROnPlateau


# In[14]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0005) 
callback = [learning_rate_reduction]


# In[15]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

print("Starting training")
final_model=model.fit_generator(train_generator_test, samples_per_epoch=100, nb_epoch=nb_epochs, verbose=1,
                    validation_data=val_generator_test,
                    nb_val_samples=nb_val_samples,callbacks=callback)

print("Initial training done, starting phase two (finetuning)")


# In[16]:


model.save('hopefinal2final.hdf5')


# In[17]:


# Load two new generator with smaller batch size, needed because using the same batch size
# for the fine tuning will result in GPU running out of memory and tensorflow raising an error
print("Loading the dataset with batch size of {}...".format(batch_size_phase_two))
train_generator_test, val_generator_test = get_training_generator_test(batch_size_phase_two)
print("Dataset loaded")


# In[18]:


# Load best weights from initial training
model.load_weights('hopefinal2final.hdf5')


# In[19]:


# Make all layers trainable for finetuning
for layer in model.layers:
    layer.trainable = True


# In[20]:


model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

history=model.fit_generator(train_generator_test, samples_per_epoch=240, nb_epoch=5, verbose=1,
                    validation_data=val_generator_test,
                    nb_val_samples=nb_val_samples,
                    callbacks=callback)


# In[21]:


model.save('cnn_rnn_67_modelfinal.hdf5')


# In[22]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[23]:


from sklearn.metrics import classification_report, confusion_matrix
import imageio
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[24]:


num_of_test_samples = 18000 
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(val_generator_test, num_of_test_samples // batch_size_phase_two,verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
matrix1 = confusion_matrix(val_generator_test.classes, y_pred)


# In[26]:


print('\nClassification Report')
target_names = ['cloudy',
                 'haze',
                 'snow',
                 'rainy',
                 'thunder',
                 'sunny']
class_report = classification_report(val_generator_test.classes, y_pred, target_names=target_names)
print(class_report)


# In[29]:


#[row, column]
TP = matrix1[1, 1]
TN = matrix1[0, 0]
FP = matrix1[0, 1]
FN = matrix1[1, 0]


# ## Classification Error: Overall, how often is the classifier incorrect?

# In[32]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(classification_error)
print(1 - metrics.accuracy_score(val_generator_test.classes, y_pred))


# In[31]:


from sklearn import metrics


# ## Sensitivity: When the actual value is positive, how often is the prediction correct?

# In[34]:


sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(val_generator_test.classes, y_pred,average='micro'))


# ## Specificity: When the actual value is negative, how often is the prediction correct?

# In[35]:


specificity = TN / (TN + FP)

print(specificity)


# ## Confusion matrix

# In[52]:


cn=metrics.confusion_matrix(val_generator_test.classes, y_pred)


# In[57]:


print(cn)


# In[79]:


import seaborn as sns
import matplotlib.pyplot as plt 
ax= plt.subplot()
sns.heatmap(cn, annot=True,fmt='g'); #annot=True to annotate cells

# labels, title and ticks
label_font = {'size':'18'}  # Adjust to fit
ax.set_xlabel('Predicted labels', fontdict=label_font);
ax.set_ylabel('True labels', fontdict=label_font);
title_font = {'size':'18'}  # Adjust to fit
ax.set_title('Confusion Matrix',fontdict=title_font); 
ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust to fit
ax.xaxis.set_ticklabels(['cloudy','haze','rainy','snow','sunny','thunder']); 
ax.yaxis.set_ticklabels(['thunder', 'sunny','snow','rainy','haze','cloudy']);


# In[65]:


from matplotlib import pyplot
import seaborn


# ## Demo 

# In[1]:


from keras.models import load_model


# In[43]:


model = load_model('cnn_rnn_68_model.hdf5')


# In[29]:


from keras.preprocessing import image
from IPython.display import Image
import numpy as np
from keras.applications import imagenet_utils
from keras.applications.resnet50 import decode_predictions


# In[38]:


import numpy as np 
import pandas as pd
import os
import json
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from IPython.display import Image, display


# In[112]:


Image(filename='D:/200X200/sunny/sunny_00139.jpg')


# In[65]:


labels=["cloudy", "haze", "rainy", "snow", "sunny", "thunder"]


# In[111]:


import numpy as np
from keras.preprocessing import image

img_width, img_height = 200,200
img = image.load_img('D:/200X200/sunny/sunny_00139.jpg', target_size = (img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
pred=model.predict(img)
pred


# In[114]:


labels[np.argmax(pred)]


# In[113]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'cloudy', 'haze', 'rainy', 'snow', 'sunny','thunder'
sizes = pred
explode = (0, 0, 0, 0,0.1,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

