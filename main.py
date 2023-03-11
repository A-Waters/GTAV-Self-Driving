from screen_grab import take_screen_shot
from process_image import process_image
import cv2
import time



cv2.imshow("",process_image(take_screen_shot()))
cv2.waitKey(0)

'''

keys = ['w','a','s','d']

import keyboard  # using module keyboardqqqqqqqqq
while True:  # making a loop
    
    collect = False
    
    if keyboard.is_pressed('['):  # if key 'q' is pressed 
        print('Starting Data Collection')
        collect = True
    
    count = 0
    while collect:
        time.sleep(0.1)
        count += 1
        shot = take_screen_shot()
        
        pressed = ""
        for key in keys:
            if keyboard.is_pressed(key):
                pressed += key

        cv2.imwrite("./images/train"+str(count)+'_'+pressed+'.jpg',shot)
        if keyboard.is_pressed(']'):
            print('Ending Data Collection')
            collect = False
'''

# import keras
# time.sleep(5)
# cv2.imshow("",take_screen_shot())
# cv2.waitKey(0)


# import tensorflow as tf
# print(tf.__version__)
# print("Num GPUs Available: ", tf.test.gpu_device_name())



'''
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
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
])
'''