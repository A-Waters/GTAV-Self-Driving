from grab_data import grab_data 
import random
from tensorflow import keras
import numpy as np
# import data
images, labels = grab_data("training_data.npz",2)

print("before", len(images))

training_data_images = []
training_data_labels = []
validation_data_images = []
validation_data_labels = []

for i, v in enumerate(images):
    if random.random() <= 0.80:
        # get training data
        training_data_images.append(images[i])
        training_data_labels.append(labels[i])
    else:
        # get validation data
        validation_data_images.append(images[i])
        validation_data_labels.append(labels[i])


print(f'total: {len(images)} training: {len(training_data_images)}:{len(training_data_labels)} validation: {len(validation_data_images)}:{len(validation_data_labels)} {len(validation_data_labels)/len(images)}%' )


training_data_images = np.asarray(training_data_images)
training_data_labels = np.asarray(training_data_labels)
validation_data_images = np.asarray(validation_data_images)
validation_data_labels = np.asarray(validation_data_labels)


# define model / # load mnodel 
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(1,270,480), data_format='channels_first',),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='softmax')
])

'''model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(480,227)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])'''


model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=["mae", "acc"], # keras.metrics.SparseCategoricalAccuracy()
)

# train model 
history = model.fit(x=training_data_images,y=training_data_labels,batch_size=32,epochs=10, validation_data=(validation_data_images, validation_data_labels))

# save model
print(history.history)

# check results