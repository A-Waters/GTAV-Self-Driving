from grab_data import grab_data 
import random
from tensorflow import keras
import tensorflow as tf
import numpy as np
import datetime
import os
import constants

# import data

print("Starting script")

images, labels = grab_data(constants.TRAINING_DATA,2)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print("ERRR:", e)



print("Sorting images")

training_data_images = []
training_data_labels = []
validation_data_images = []
validation_data_labels = []

for i, v in enumerate(images):
    if random.random() <= constants.TRAINING_VALIDATION_SPLIT:
        # get training data
        training_data_images.append(images[i])
        training_data_labels.append(labels[i])
    else:
        # get validation data
        validation_data_images.append(images[i])
        validation_data_labels.append(labels[i])


print(f'total: {len(images)} training: {len(training_data_images)}:{len(training_data_labels)} validation: {len(validation_data_images)}:{len(validation_data_labels)} {len(validation_data_labels)/len(images)}%' )


training_data_images = np.asarray(training_data_images)
training_data_labels = np.asarray(training_data_labels).squeeze()
validation_data_images = np.asarray(validation_data_images)
validation_data_labels = np.asarray(validation_data_labels).squeeze()


# define model / # load mnodel 
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(1,270,480), data_format='channels_first',),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='softmax')
])

print("Building model")
'''
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu', input_shape=(1,constants.IMAGE_HEIGHT,constants.IMAGE_WIDTH), data_format='channels_first',),
    keras.layers.Flatten(),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])'''

print("Comipleing model")

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.CategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=["mae", "acc"], # keras.metrics.SparseCategoricalAccuracy()
)



checkpoint_dir = os.path.dirname(constants.CHECKPOINT_PATH)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=constants.CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=constants.SAVE_FREQ*constants.BATCH_SIZE)



log_dir = constants.LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tf.compat.v1.summary.FileWriterCache.clear()


print("Training")

# train model 
history = model.fit(x=training_data_images,y=training_data_labels,batch_size=constants.BATCH_SIZE,epochs=constants.EPOCHS, validation_data=(validation_data_images, validation_data_labels), callbacks=[tensorboard_callback])

# save model
print(history.history)

# check results