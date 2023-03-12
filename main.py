from screen_grab import take_screen_shot
from process_image import process_image
import cv2
import time
from tensorflow import keras
import tensorflow as tf

print(tf.config.experimental.list_physical_devices())


cv2.imshow("",process_image(take_screen_shot()))
print(process_image(take_screen_shot()))
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
])
'''

'''



def get_model(data_in, data_out, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_x = spec_start
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters = cnn_nb_filt, kernel_size=(2, 2), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)
        spec_x = Permute((2, 1, 3))(spec_x)
        spec_x = Reshape((data_in.shape[-2], -1))(spec_x)for _r in _rnn_nb:
        spec_x = Bidirectional(
        GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
        merge_mode='concat')(spec_x)for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
        out = Activation(‘sigmoid', name='strong_out')(spec_x)_model = Model(inputs=spec_start, outputs=out)
    _model.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
                                                                        
    _model.summary()
 return _model
'''