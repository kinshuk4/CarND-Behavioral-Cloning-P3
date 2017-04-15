import os
import csv
import cv2

import model_list as ml
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import data_util as du


def main():
    data_subfolder_ids = [2]
    images, measurements = du.get_images_measurements(data_subfolder_ids)
    print("Image Samples: " + str(len(images)))
    # model_type="simple"
    # model_type="lenet"
    model_type="nvidia"
    print("Getting the model.........................."+model_type + "\n")
    model, epochs = ml.choose_model(model_type)

    # Convertint to numpy format as Keras requires
    X_train = np.array(images)
    y_train = np.array(measurements)

    import model_util as mu
    mu.compile_model(model, X_train, y_train, epochs)


if __name__ == "__main__":
    main()
