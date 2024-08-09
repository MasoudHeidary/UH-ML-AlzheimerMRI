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
    l.println(f"TPU device: {tpu.master()}")
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
l.println(f"tensorflow version: {tf.__version__}")


# basic configuration
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [176, 208]
EPOCHS = 10


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
if False:
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


def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

NUM_IMAGES = []

for label in class_names:
    dir_name = "./Alzheimer_s Dataset/train/" + label[:-2] + 'ed'
    NUM_IMAGES.append(len([name for name in os.listdir(dir_name)]))

l.println(f"#images: {NUM_IMAGES}")


### CNN
def conv_block(filters):
    block = tf.keras.Sequential(
        [
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
        ]
    )
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    return block

def build_model():
    model = tf.keras.Sequential(
        [
        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ]
    )
    return model
### END CNN

with strategy.scope():
    model = build_model()
    METRICS = [tf.keras.metrics.AUC(name='auc')]
    
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=METRICS
    )


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("alzheimer_model.keras", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=EPOCHS
)



# plot the accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
ax = ax.ravel()

for i, met in enumerate(['auc', 'loss']):
    ax[i].plot(history.history[met], linewidth=5)
    ax[i].plot(history.history['val_' + met], linewidth=5)

    ax[i].set_title('Model {}'.format(met), fontsize=28, fontweight='bold')
    ax[i].set_xlabel('epochs', fontsize=28, fontweight='bold')
    ax[i].set_ylabel(met, fontsize=28, fontweight='bold')

    ax[i].legend(['train', 'val'], fontsize=28)
    plt.xticks(fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')
plt.show()



test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./Alzheimer_s Dataset/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Evaluate the model on the test set
test_loss, test_auc = model.evaluate(test_ds)
l.println(f"Test Loss: {test_loss:.4f}")
l.println(f"Test AUC: {test_auc:.4f}")



# # Function to get true labels and predictions
# def get_labels_and_predictions(dataset):
#     true_labels = []
#     predictions = []
#     images_list = []
#     for images, labels in dataset:
#         preds = model.predict(images)
#         true_labels.extend(labels.numpy())
#         predictions.extend(preds)
#         images_list.extend(images.numpy())
#     return np.array(true_labels), np.array(predictions), np.array(images_list)

# # Get true labels, predictions, and images for the test set
# true_labels, predictions, images_list = get_labels_and_predictions(test_ds)

# # Convert one-hot encoded labels to class indices
# true_labels_indices = np.argmax(true_labels, axis=1)
# predictions_indices = np.argmax(predictions, axis=1)

# # Display the first 50 predictions and true labels
# plt.figure(figsize=(20, 20))
# for i in range(50):
#     ax = plt.subplot(10, 5, i + 1)
#     plt.imshow(images_list[i].astype("uint8"))
#     plt.title(f"True: {class_names[true_labels_indices[i]]}\nPred: {class_names[predictions_indices[i]]}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

