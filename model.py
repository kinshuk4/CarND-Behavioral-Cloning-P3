import os
import csv
import cv2

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np


def main():
    numSubFolder=[2]
    allLines = readAllLogs(numSubFolder)
    
    print("Total Samples: "+ str(len(allLines)))

    images = []
    measurements = []
    print("Readying the data set..........................\n")
    for line in allLines:
        #print(line)
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(float(line[3]))

    # Convertint to numpy format as Keras requires
    X_train = np.array(images)
    y_train = np.array(measurements)

    from keras.models import Sequential, Model
    from keras.layers import Flatten, Dense, Lambda

    print("Training the model..........................\n")
    model = Sequential()
    # https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/b3883cd8-f915-46e1-968a-e935323e749b
    #In this project, a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py.
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    print("Compiling the model..........................\n")
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    print("Saving the model..........................\n")
    model.save('model.h5')



def readAllLogs(numSubFolder):
    dataFolder="../DataSets/carnd-behavioral-cloning-p3-data/data"
    allLines=[]
    
    for i in numSubFolder:
        datapath=dataFolder+str(i)
        csvFile = datapath+"/driving_log.csv"
        imageFolder=datapath+"/IMG/"
        lines=readLogs(csvFile, imageFolder)
        allLines.extend(lines)
    return allLines
    
'''
data directory has a date folder and then the csv file and IMG folder.
'''
def readLogs(csvFile, imageFolder):
    lines=[]
    with open(csvFile) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            #Pointing the image path to the right path.
            # Sample line: ['/Users/kchandra/Desktop/images/IMG/center_2017_04_10_00_22_50_654.jpg', '/Users/kchandra/Desktop/images/IMG/left_2017_04_10_00_22_50_654.jpg', '/Users/kchandra/Desktop/images/IMG/right_2017_04_10_00_22_50_654.jpg', '0', '0', '0', '23.57149']
            # line[0],line[1] and line[2] are center, left and right view of images
            line[0]=getCurrFilePath(line[0], imageFolder)
            line[1]=getCurrFilePath(line[1], imageFolder)
            line[2]=getCurrFilePath(line[2], imageFolder)
            lines.append(line)
    return lines


def getCurrFilePath(source_path, current_path):
    filename = source_path.split('/')[-1]
    return current_path+filename

if __name__ == "__main__":
    main()
