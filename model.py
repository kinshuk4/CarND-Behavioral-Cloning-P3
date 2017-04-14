import os
import csv
import cv2

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np


def main():
    numSubFolder=2
    allLines = readAllLogs(numSubFolder)
    
    print("Total Samples: "+ str(len(allLines)))

    images = []
    measurements = []
    print("Readying the data set..........................\n")
    for line in allLines:
        img_center = cv2.imread(line[0])
        images.append(img_center)
        measurements.append(float(line[3]))

    X_train = np.array(images)
    y_train = np.array(measurements)

    from keras.models import Sequential, Model
    from keras.layers import Flatten, Dense

    print("Training the model..........................\n")
    model = Sequential()
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
    
    for i in range(1, numSubFolder):
        print(dataFolder)
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
        for line in reader:
            print(line)
            #Pointing the image path to the right path.
            # Sample line: ['/Users/kchandra/Desktop/images/IMG/center_2017_04_10_00_22_50_654.jpg', '/Users/kchandra/Desktop/images/IMG/left_2017_04_10_00_22_50_654.jpg', '/Users/kchandra/Desktop/images/IMG/right_2017_04_10_00_22_50_654.jpg', '0', '0', '0', '23.57149']
            # line[0],line[1] and line[2] are center, left and right view of images
            line[0]=getCurrFilePath(line[0],imageFolder)
            line[1]=getCurrFilePath(line[1],imageFolder)
            line[2]=getCurrFilePath(line[2],imageFolder)
            
            lines.append(line)
    return lines


def getCurrFilePath(source_path, current_path):
    filename = source_path.split('/')[-1]
    return current_path+filename

if __name__ == "__main__":
    main()
