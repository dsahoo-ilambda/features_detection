#!/usr/bin/env python
# coding: utf-8

import pathlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import tensorflow as tf
import re
import pathlib
import glob
import os
import time
import json
import logging.handlers

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
RGB_IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
BATCH_SIZE = 32

keras = tf.keras

TRAIN_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_train_split/whole_resize'
TEST_IMAGE_PATH = '/home/ilambda/goods_viewer/Debasish/dataset/1_eval_img_resize/'
MODEL_NAME = "attribute_detection_2"
LOG_FILE_NAME = MODEL_NAME + "_log.log"
FORMAT = '%(asctime)s [%(levelname)s] %(message)s'


# ### Data analysis tasks
# 
# - Create a dataframe of image_name to labels input file (labels_map_df)
# - Import the features file as a list (features_list)
# - Filter the dataframe columns to retain all the columns matching in features_file and discard the rest. Copy the resulting dataframe to a new dataframe (updated_labels_df)
#     - Make sure to add the columns ['#Attr 266', 'Name', 'Price']
#     
# Create the following plots
# 
# -


def configure_logger():
    # Initiate logging
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__file__)
    # Add a file logger
    file_handler = logging.handlers.RotatingFileHandler(filename=LOG_FILE_NAME)
    file_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(file_handler)
    # without this, logger doesn't work
    logger.setLevel(logging.INFO)

    return logger


def write_file_to_disk(file_name, contents):
    with open(file_name, 'w') as output:
        output.write(contents)
    logger.info(f"Model history written to file: {file_name}")


def generate_file_name(name, type="--"):
    return name + "_" + type + "_" + time.strftime("%Y_%m_%d_%H_%M_%S")


logger = configure_logger()

labels = pd.read_csv('dataset/jc_input.txt')
labels.head()
logger.info(f"Shape of the dataset: {labels.shape}")


features = np.loadtxt('dataset/selected_feature.txt', dtype=str, delimiter='\n')
logger.info(f"Number of features in features.txt: {len(features)}")

# ### Change the column names in dataset. Repelace pipe with space.
# This is done to match the column names specified in  the selected_features.txt
columns_with_pipe = list(filter(lambda x: re.match('.*\|.*', x), labels.columns))
len(columns_with_pipe)
columns_replaced_pipe = set(map(lambda x: x.replace('|', ' '), labels.columns))

len((columns_replaced_pipe).intersection(set(features)))

# ### List of columns containing the labels
# Sorted alphabetically
label_columns = sorted(list((columns_replaced_pipe).intersection(set(features))))

NUM_CATEGORIES = len(label_columns)

all_columns = ["#Attr 266", "Name", "Price"] + label_columns

updated_labels_df = labels.copy()

# ### Stores the mapping of renamed_cols to original_cols

renamed_to_orig_cols_dict = dict([(col, col.replace('|', ' ')) for col in updated_labels_df.columns])

updated_labels_df.rename(renamed_to_orig_cols_dict, axis=1, inplace=True)

updated_labels_df = updated_labels_df[all_columns]

updated_labels_df.shape

# ### Update the column name "#Attr 266" to filename
updated_labels_df.rename({'#Attr 266': "filename"}, axis=1, inplace=True)
updated_labels_df["filename"] = updated_labels_df["filename"].apply(lambda x: str(x) + ".jpg")

training_list = [f.name for f in pathlib.Path(TRAIN_IMAGE_PATH).glob('*.jpg')]
testing_list = [f.name for f in pathlib.Path(TEST_IMAGE_PATH).glob('*.jpg')]

# In[29]:


training_fname = list(set(updated_labels_df.filename.values).intersection(set(training_list)))
testing_fname = list(set(updated_labels_df.filename.values).intersection(set(testing_list)))
logger.info(f"Number of training files with attributes: {len(training_fname)}")
logger.info(f"Number of testing files with attributes: {len(testing_fname)}")

# ### Remove duplicates

# In[30]:


# Identify duplicate files
updated_labels_df.filename.value_counts()

# In[31]:


# 4 files with name '&  ress.jpg' are duplicated
to_remove_index = updated_labels_df[(updated_labels_df.filename == '&  ress.jpg')].index

# In[32]:


updated_labels_df.drop(to_remove_index, inplace=True)

# ### Remove nans
rows_with_nan = updated_labels_df[updated_labels_df[label_columns].isna().sum(axis=1) > 0].index
logger.info(f"Number of rows with NaN: {len(rows_with_nan)}")
updated_labels_df.drop(rows_with_nan, inplace=True)
updated_labels_df.shape

# Check for further nans
sum(updated_labels_df[label_columns].isna().sum() > 0)
# ### Update the Nan name attribute with blank_string
updated_labels_df.Name.fillna("", inplace=True)

# ### Train test split
# - update index to dataframe to file name
# - perform a set intersection of df index and training_fname from the training images directory
# - perform a set intersection of df index and testing_fname from the testing images directory
# - create separate dfs - training_df and testing_df

updated_labels_df.set_index(updated_labels_df.filename, inplace=True)

train_index = set(updated_labels_df.index).intersection(set(training_fname))
test_index = set(updated_labels_df.index).intersection(set(testing_fname))

# ### Training and testing df
training_df = updated_labels_df.loc[list(train_index)]
testing_df = updated_labels_df.loc[list(test_index)]

logger.info(f"Training df: {training_df.shape}")
logger.info(f"Testing df: {testing_df.shape}")

# ### Create generators

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.3)
testdatagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# use class_mode as 'multi_output' or 'other'

# for other - each label would be 229 dimensions
training_gen = datagen.flow_from_dataframe(training_df,
                                           directory=TRAIN_IMAGE_PATH,
                                           x_col='filename',
                                           y_col=label_columns,
                                           class_mode='other',
                                           target_size=IMAGE_SIZE,
                                           subset='training',
                                           shuffle=False)

validation_gen = datagen.flow_from_dataframe(training_df,
                                             directory=TRAIN_IMAGE_PATH,
                                             x_col='filename',
                                             y_col=label_columns,
                                             class_mode='other',
                                             target_size=IMAGE_SIZE,
                                             subset='validation',
                                             shuffle=False)

test_gen = testdatagen.flow_from_dataframe(testing_df,
                                           directory=TEST_IMAGE_PATH,
                                           x_col='filename',
                                           y_col=label_columns,
                                           class_mode='other',
                                           target_size=IMAGE_SIZE,
                                           shuffle=False)



# ### Custom model creation
# - Sequential() model
# - Activation at prediction layer = sigmoid, NUM_CATEGORIES = 229
#     - sigmoid will return a score in (0,1) for each feature
#     - in total 229 features will be returned in form of a vector
# - Loss function is binary_crossentropy
#     - loss function calculates the loss by comparing the predicted_loss by sigmoid and the actual label (0 or 1) of the feature
"""
base_model = keras.applications.resnet.ResNet50(include_top=False,
                                                weights='imagenet',
                                                input_shape=RGB_IMAGE_SIZE,
                                                pooling='avg')
"""

base_model_name = 'ecom-image-model_basemodel_2020_02_20_20_25_21.h5'
logger.info(f"Loading base model: {base_model_name}")
base_model = keras.models.load_model(base_model_name)
base_model.trainable = False
logger.info(f"Base Model trainable = {base_model.trainable}")

model = keras.models.Sequential(name=MODEL_NAME)
model.add(base_model)
model.add(keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

EPOCHS = 30
logger.info(f"Total number of epochs: {EPOCHS}")

tensorboard = keras.callbacks.TensorBoard(
  log_dir='./logs',
  histogram_freq=1,
  write_images=True
)


history = model.fit(training_gen, validation_data=validation_gen, verbose=1, epochs=EPOCHS, shuffle=False, callbacks=[tensorboard])
model_name = generate_file_name(model.name, type="model") + ".h5"
logger.info(f"Saving the model as : {model_name}")
model.save(model_name)

# Save history to disk
file_contents = json.dumps(str(history.history))
write_file_to_disk(generate_file_name(model.name, type="history") + ".json", file_contents)
# model.save("attribute_detection.h5")
logger.info(history.history)
logger.info(f"Training complete")
