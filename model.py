import os
import csv
import cv2

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential, Model

def main():
    lines=[]
    numSubFolder=2
    allLines = readAllLogs(numSubFolder)
    print("Total Samples: "+len(allLines))

    for line in allLines:
        img_center=cv2.imread(line[0].replace("/Users/kchandra/Desktop/images/",imageFolder))


def readAllLogs(numSubFolder):
    dataFolder="../DataSets/carnd-behavioral-cloning-p3-data/data"
    allLines=[]
    
    for i in range(1, numSubFolder):
        print(dataFolder)
        datapath=dataFolder+str(i)
        csvFile = datapath+"/driving_log.csv"
        imageFolder=datapath+"/IMG/"
        lines=readLogs(csvFile, imageFolder)
        allLines.append(lines)
'''
data directory has a date folder and then the csv file and IMG folder.
'''
def readLogs(csvFile, imageFolder):
    lines=[]
    with open(csvFile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            print(line)
            lines.append(line)
    return line

if __name__ == "__main__":
    main()
