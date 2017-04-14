import os
import csv
import cv2
import numpy as np



def read_all_logs(numSubFolder):
    dataFolder = "../DataSets/carnd-behavioral-cloning-p3-data/data"
    allLines = []
    for i in numSubFolder:
        datapath = dataFolder + str(i)
        csvFile = datapath + "/driving_log.csv"
        imageFolder = datapath + "/IMG/"
        lines = read_logs(csvFile, imageFolder)
        allLines.extend(lines)
    return allLines


'''
data directory has a date folder and then the csv file and IMG folder.
'''
def read_logs(csvFile, imageFolder):
    lines = []
    with open(csvFile) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            # Pointing the image path to the right path.
            # Sample line: ['/Users/kchandra/Desktop/images/IMG/center_2017_04_10_00_22_50_654.jpg', '/Users/kchandra/Desktop/images/IMG/left_2017_04_10_00_22_50_654.jpg', '/Users/kchandra/Desktop/images/IMG/right_2017_04_10_00_22_50_654.jpg', '0', '0', '0', '23.57149']
            # line[0],line[1] and line[2] are center, left and right view of images
            line[0] = get_current_file_path(line[0], imageFolder)
            line[1] = get_current_file_path(line[1], imageFolder)
            line[2] = get_current_file_path(line[2], imageFolder)
            lines.append(line)
    return lines


def get_current_file_path(source_path, current_path):
    filename = source_path.split('/')[-1]
    return current_path + filename