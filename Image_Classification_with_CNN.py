#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# <h1 style="text-align: center;">Deep Learning<br><br>Assignment-2 (CNN)<br><br>Image Classification with CNN<br><h1>

# # Task and Dataset Info
# 
# Welcome to second assignment of Deep learning lesson. Follow the instructions and complete the assignment.
# 
# **Build an image classifier with Convolutional Neural Networks for the Fashion MNIST dataset. This data set includes 10 labels of different clothing types with 28 by 28 *grayscale* images. There is a training set of 60,000 images and 10,000 test images.**
# 
#     Label	Description
#     0	    T-shirt/top
#     1	    Trouser
#     2	    Pullover
#     3	    Dress
#     4	    Coat
#     5	    Sandal
#     6	    Shirt
#     7	    Sneaker
#     8	    Bag
#     9	    Ankle boot

# # Import Libraries

# In[1]:


try:
    import jupyter_black
    jupyter_black.load()
except ImportError:
    print("You can safely ignore this message.")


# In[2]:


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# In[3]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
from tensorflow import keras

# import warnings
# warnings.filterwarnings("ignore")
# warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10, 6)

sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Set it None to display all rows in the dataframe
# pd.set_option('display.max_rows', None)

# Set it to None to display all columns in the dataframe
pd.set_option("display.max_columns", None)


# In[4]:


import tensorflow as tf

if tf.config.list_physical_devices("GPU"):
    print("GPU support is enabled for this session.")
else:
    print("CPU will be used for this session.")


# In[5]:


# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) `tensorflow` random seed
# 3) `python` random seed
SEED = 42
keras.utils.set_random_seed(SEED)

# This will make TensorFlow ops as deterministic as possible, but it will
# affect the overall performance, so it's not enabled by default.
# `enable_op_determinism()` is introduced in TensorFlow 2.9.
tf.config.experimental.enable_op_determinism()


# # Recognizing and Understanding Data
# 
# **TASK 1: Run the code below to download the dataset using Keras.**

# In[6]:


from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[7]:


print(f"There are {len(x_train)} images in the training dataset")
print(f"There are {len(x_test)} images in the test dataset")


# In[8]:


x_train[5].shape


# **TASK 2: Use matplotlib to view an image from the data set. It can be any image from the data set.**

# In[9]:


classes=["T-shirt/top", "Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# In[10]:


import matplotlib.pyplot as plt

# Choose an image
index = 97 

# Display the selected image along with its class name
plt.figure()
plt.imshow(x_train[index], cmap='gray')
plt.title(f'Class: {classes[y_train[index]]}')
plt.show()


# # Data Preprocessing
# 
# **TASK 3: Normalize the X train and X test data by dividing by the max value of the image arrays.**

# In[11]:


max_pixel_value = np.max(x_train)

# Normalize the data by dividing by the maximum pixel value
x_train = x_train / max_pixel_value
x_test = x_test / max_pixel_value
max_pixel_value


# **Task 4: Reshape the X arrays to include a 4 dimension of the single channel. Similar to what we did for the numbers MNIST data set.**

# In[12]:


# Reshape the X arrays to include the single channel
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# **TASK 5: Convert the y_train and y_test values to be one-hot encoded for categorical analysis by Keras.**

# In[13]:


from tensorflow.keras.utils import to_categorical


# In[14]:


# Convert y_train and y_test to one-hot encoded format
y_train = to_categorical(y_train, num_classes=10)  
y_test = to_categorical(y_test, num_classes=10)


# # Modeling

# ## Create the model
# 
# **TASK 5: Use Keras to create a model consisting of at least the following layers (but feel free to experiment):**
# 
# * 2D Convolutional Layer, filters=28 and kernel_size=(3,3)
# * Pooling Layer where pool_size = (2,2) strides=(1,1)
# 
# * Flatten Layer
# * Dense Layer (128 Neurons, but feel free to play around with this value), RELU activation
# 
# * Final Dense Layer of 10 Neurons with a softmax activation
# 
# **Then compile the model with these parameters: loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']**

# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create the model
model = Sequential()

# 2D Convolutional Layer, filters=28 and kernel_size=(3,3)
model.add(Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Pooling Layer where pool_size = (2,2) strides=(1,1)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten Layer
model.add(Flatten())

# Dense Layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Final Dense Layer of 10 Neurons with a softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model with the specified parameters
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ##  Model Training 
# 
# **TASK 6: Train/Fit the model to the x_train set by using EarlyStop. Amount of epochs is up to you.**

# In[16]:


from tensorflow.keras.callbacks import EarlyStopping


# In[17]:


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# In[18]:


history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])


# **TASK 7: Plot values of metrics you used in your model.**

# In[19]:


# Get training and validation loss and accuracy
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# plot for the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plot for the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# ## Model Evaluation
# 
# **TASK 8: Show the accuracy,precision,recall,f1-score the model achieved on the x_test data set. Keep in mind, there are quite a few ways to do this, but we recommend following the same procedure we showed in the MNIST lecture.**

# In[20]:


from sklearn.metrics import classification_report


# In[21]:


# Predict the classes on the x_test data
y_pred = model.predict(x_test)


# In[22]:


# Convert the one-hot encoded predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)


# In[23]:


# Convert one-hot encoded ground truth labels to class labels
y_test_labels = np.argmax(y_test, axis=1)


# In[24]:


# Calculate the classification report
report = classification_report(y_test_labels, y_pred_labels)

print(report)


# In[25]:


unique, counts = np.unique(y_test_labels, return_counts=True)

# Create a dictionary to display the count of data samples for each class
class_count = dict(zip(unique, counts))

# Print the count of data samples for each class
for class_label, count in class_count.items():
    print(f"Class {class_label}: {count} samples")


# ## Prediction

# In[53]:


new_image = x_test[54]


# In[54]:


new_image.shape


# In[55]:


plt.imshow(new_image)
plt.show()


# In[56]:


image_prediction = model.predict(new_image.reshape(1, 28, 28, 1))


# In[57]:


predicted_label = np.argmax(image_prediction)

# Display the predicted class label
print(f"Predicted Label: {classes[int(predicted_label)]}")


# # End of Assignment

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
