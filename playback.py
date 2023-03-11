import numpy as np
import cv2
import time

with open("training_data.npz", 'rb') as f:
    data = np.load(f)
    d = dict(zip(("data1A","data1B"), (data[k] for k in data)))
    images = d["data1A"].squeeze()
    labels = d["data1B"].squeeze()
    print(images.shape)
    for i, v in enumerate(images):
        print(labels[i])
        cv2.imshow("", v)
        cv2.waitKey(100)