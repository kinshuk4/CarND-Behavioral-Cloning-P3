import os
import csv
import cv2
import numpy as np
import sklearn

def flip_image(image, measurement):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped


def augment_data(images, measurements):
    augmented_images = []
    augmented_measurements = []

    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)

        image_flipped, measurement_flipped = flip_image(image, measurement)
        augmented_images.append(image_flipped)
        augmented_measurements.append(measurement_flipped)
    return augmented_images, augmented_measurements


def process_line(line, images, measurements):
    img_center = cv2.imread(line[0])
    img_left = cv2.imread(line[1])
    img_right = cv2.imread(line[2])

    images.append(img_center)
    images.append(img_left)
    images.append(img_right)

    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

def get_images_measurements(data_subfolder_ids):
    all_lines = read_all_logs(data_subfolder_ids)
    print("Total Samples: " + str(len(all_lines)))
    images = []
    measurements = []
    print("Readying the data set..........................\n")
    for line in all_lines:
        process_line(line, images, measurements)

    augmented_images, augmented_measurements = augment_data(images, measurements)
    return augmented_images, augmented_measurements


def read_all_logs(sub_folder_ids):
    dataFolder = "../DataSets/carnd-behavioral-cloning-p3-data/data"
    allLines = []
    for i in sub_folder_ids:
        datapath = dataFolder + str(i)
        print("Reading the data folder: "+datapath)
        csv_file = datapath + "/driving_log.csv"
        image_folder = datapath + "/IMG/"
        lines = read_logs(csv_file, image_folder)
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


def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                process_line(batch_sample, images, measurements)

            augmented_images, augmented_measurements = augment_data(images, measurements)
            # trim image to only see section with road
            # Convert array to numpy format as Keras requires
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)