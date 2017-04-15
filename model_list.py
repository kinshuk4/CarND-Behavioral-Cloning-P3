from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


'''
Get the model based on @model_type
'''
def choose_model(model_type):
    # choose model
    epochs = 0
    if (model_type == 'simple'):
        model = get_sequential_model()
        epochs = 7
    elif (model_type == 'lenet'):
        model = get_lenet_model()
        epochs = 5

    return model, epochs

'''
n this project, a lambda layer is a convenient way to parallelize image normalization. 
The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py. 
More: https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/b3883cd8-f915-46e1-968a-e935323e749b
'''
def get_normalized_cropped_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # The example below crops:
    #
    # 50 rows pixels from the top of the image
    # 20 rows pixels from the bottom of the image
    # 0 columns of pixels from the left of the image
    # 0 columns of pixels from the right of the image
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    return model

def get_simple_model():
    # SEQUENTIAL SIMPLE
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

'''
LENET
Original LeNet takes 32X32X1 image, but here we have 160X320X3 image. 
So, convolutional networks will help
'''
def get_lenet_model():
    model = get_normalized_cropped_model()

    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


'''
NVIDIA MODEL
'''
def get_nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # crop image to only see section with road
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model