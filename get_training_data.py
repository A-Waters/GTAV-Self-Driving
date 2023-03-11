from get_keys import key_check
from screen_grab import take_screen_shot
from process_image import process_image
import cv2
import numpy as np
import time
import os

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

image_file_name = "training_data.npz"


def capture():
    if os.path.isfile(image_file_name):
        saved_data = np.load(image_file_name)
        out_data = dict(zip(("images","labels"), (saved_data[k] for k in saved_data)))
        
        training_image_data = out_data['images']
        training_label_data = out_data['labels']
        print("using exsiting data")

    else:
        training_image_data = []
        training_label_data = []
        print("new data creating")
    
    while True:
        time.sleep(0.1)
        
        image = take_screen_shot()
        out_image = process_image(image)
        print(out_image.shape)
        key_output = process_keys(key_check())
        
        training_image_data.append([out_image])
        
        training_label_data.append([key_output])
        print(key_output)

        if len(training_image_data) % 100 == 0:
            print(len(training_image_data))
            # cv2.imshow("",out_image)
            # cv2.waitKey(0)
            with open(image_file_name, 'wb') as f:
                np.savez(f, training_image_data, training_label_data)

        if 'P' in key_check():
            break
        
    


if __name__ == "__main__":
    print("start")
    capture()