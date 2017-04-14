import os
import csv
import cv2

import model_list as ml
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np






def main():
    data_subfolder_ids = [2]
    all_lines = read_all_logs(data_subfolder_ids)

    print("Total Samples: " + str(len(all_lines)))

    images = []
    measurements = []
    print("Readying the data set..........................\n")
    for line in all_lines:
        # print(line)
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(float(line[3]))

    print("Getting the model..........................\n")
    # model, epochs = ml.choose_model("sequential")
    model, epochs = ml.choose_model("lenet")

    # Convertint to numpy format as Keras requires
    X_train = np.array(images)
    y_train = np.array(measurements)

    import model_util as mu
    mu.compile_model(X_train, epochs, model, y_train)





if __name__ == "__main__":
    main()
