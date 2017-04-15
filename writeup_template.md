#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
Following are the steps I followed:
* To begin with I used the keras sequential model. It was not of much help as error was more than 400000%.
* Then added the normalization layer, which improved the model, but error was 1682 (i.e. 168200 %)
* Error was very high, and my car was swirling in circle and going into water. So, the friend LeNet came into pictures. It reduced the validation_error to .0123.
* Later I augmented the data, and this gave the *best result* with validation error of 0.0113. But this had one error, that error was less than validation_error showing signs of overfitting. 
* I then took the left and right camera images as well, with the correction of 0.2.  0.0224.
* Then I switched to Nvidia model, which made the error to 0.0191
* Finally used generators which reduce the time of training, but validation error was 0.0190. The training loss was 0.0193 which is more than validation loss, showing model is not overfitted.
 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.  (model_list.py lines 80). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 12). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). Also, each model required different number of epochs. So, when choosing the model from `choose_model` (model_list lines 10) I was returning the required number of epochs observed during each model.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I have created the separate [DataSets](https://github.com/kinshuk4/DataSets/tree/master/carnd-behavioral-cloning-p3-data) repo to contain the data. So, in this repo, there is a directory carnd-behavioral-cloning-p3-data. data1 is the data which I generated when running the simulator. data2 folder contains the images provided [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). Also, I am planning to create data3 directory to contain the data for track2. Which I will do in some days. This is how I increased the data 6x if x were the number of images:
* I augmented the data by flipping the images and also the corresponding model. Total = 2x images
* I used the left and right camera images with correction of +0.2 and -0.2 respectively.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use to convolutional network model and LeNet and convnet mentioned in Nvidia paper really helped. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with ratio 0.8 to 0.2. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To overcome this, we need to generalize a model a lot. To do so, I increased the data:
* Validation and Training set were broken
* I used my simulator data as well as one provided and augmented it by flipping the image left to right and also multiplied measurements by -1. 
* I used the left and right camera images with correction of +0.2 and -0.2 respectively.
* Epochs were tuned according to the model. 


####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes :
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][/writeup_images/center_2016_12_01_13_30_48_287.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][/writeup_images/center_2016_12_01_13_30_48_287.jpg]
![alt text][/writeup_images/left_2016_12_01_13_30_48_287.jpg]
![alt text][/writeup_images/right_2016_12_01_13_30_48_287.jpg]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][/writeup_images/left_2016_12_01_13_30_48_287.jpg]
![alt text][/writeup_images/flipped_left_2016_12_01_13_30_48_287.jpg]



After the collection process, I had 6*24111 number of data points. I then preprocessed this data by:
* Randomly choose center, left or right camera images
* add +0.2 to steer for left camera image
* add -0.2 to steer for right camera image


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
