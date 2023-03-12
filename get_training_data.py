from get_keys import key_check
from screen_grab import take_screen_shot
from process_image import process_image
import cv2
import numpy as np
import time
import os
import constants
import grab_data


def process_keys(keys):

    #       w a s d 
    code = [0,0,0,0]

    if 'W' in keys:
        code[0] = 1
    
    if 'A' in keys:
        code[1] = 1
    
    if 'S' in keys:
        code[2] = 1
    
    if 'D' in keys:
        code[3] = 1


    return code




def capture():
    
    training_image_data, training_label_data = grab_data.grab_data(constants.TRAINING_DATA,2)
    training_image_data = list(training_image_data)
    training_label_data = list(training_label_data)

    capture = False

    while True:
        if 'O' in key_check():
            capture = True
            print("starting data collection")
            time.sleep(5)


        while capture:
            time.sleep(0.1)
            
            image = take_screen_shot()
            out_image = process_image(image)

            key_output = process_keys(key_check())
            
            training_image_data.append([out_image])
            
            training_label_data.append([key_output])


            if len(training_image_data) % 500 == 0:
                print(len(training_image_data))
                with open(constants.TRAINING_DATA, 'wb') as f:
                    np.savez(f, training_image_data, training_label_data)

            if 'P' in key_check():
                capture = False
                print("ending data collection")
        
    


if __name__ == "__main__":
    print("start")
    capture()