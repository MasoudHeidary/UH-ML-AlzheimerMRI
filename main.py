import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# import PIL


# initialize log library
import log
l = log.Log()
l.println("== START ==")


# detect and configure the TPU OR get back to a default strategy using
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)


# basic configuration
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [176, 208]
EPOCHS = 250


# loading train and test data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./Alzheimer_s Dataset/train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./Alzheimer_s Dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# label classes
class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
train_ds.class_names = class_names
val_ds.class_names = class_names

NUM_CLASSES = len(class_names)

# plot an example of the labels
if True:
    plt.figure(figsize=(13, 10))
    for images, labels in train_ds.take(1):

        def plot_img(sub, index):
            ax = plt.subplot(1, 4, sub)
            plt.imshow(images[index].numpy().astype("uint8"))
            plt.title(train_ds.class_names[labels[index]])
            plt.axis("off")

        plot_img(sub=1, index=2)        # non dementia
        plot_img(sub=2, index=1)        # very mild dementia
        plot_img(sub=3, index=4)        # mild dementia
        plot_img(sub=4, index=0)        # moderate dementia

    plt.show()
