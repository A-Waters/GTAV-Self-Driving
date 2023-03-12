import grab_data
from collections import Counter
import constants
import numpy as np

print("Starting")
images, labels = grab_data.grab_data(constants.TRAINING_DATA,2)

uniques = {}
for label in labels:
    found = False

    for uniq in uniques.keys():
        if np.array_equal(tuple(label[0]),uniq):
            uniques[tuple(label[0])] += 1
            found = True
    

    if found == False:
        uniques[tuple(label[0])] = 1
        print(label[0])
            

print(uniques)