# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/fail_train1.png "Model Visualization"
[image2]: ./writeup_images/Figure_1-1.png "Grayscaling"
[image3]: ./writeup_images/issuearea.png "Recovery Image"
[image4]: ./writeup_images/training.png "Recovery Image"
[image5]: ./writeup_images/canny.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with an input of 1 convolutional layer, followed by a dropout layer with a 40% dropout rate, a max pooling layer, and finally a fully connected layer with the steering command as output.

I had attempted to use the same architecture used by the autonomous vehicle team at NVIDIA. This architecture proved to achieve a solution very well for some, but I could not tune the parameters well enough for proper training, so the model explained above was used.

The model includes a RELU activations after the convolution to introduce nonlinearity. After the convolutional layers, the fully connected layer provide the network the ability to merge the features from the convolutions. The final layer is a `Dense()` fully connected layer with an output size of 1 which is the steering command. This steering command is used to control the vehicle.

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, training was first conducted with 14 epochs which resulted in the image below:

![alt text][image1]

This allowed me to see that I should reduce the number of epochs to 6. After 6 epochs, the model seems to overfit since the validation mse spikes and the training mse continues to decrease. I did make some attempts to add dropout layers in the model to reduce overfitting, but these additions continued to reduce the accuracy of the model, so I decided to not include them.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 142).

#### 4. Appropriate training data

The training data that I used was just the data that was collected by the Udacity team. This data was sufficient enough to train my vehicle. Other data was collected, but not utilized in the final model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the well defined model from NVIDIA and improve smaller parameters/data augmentation to achieve a full lap. This was attempted but this model seem to be too much for this application. So another model was used which will be explained in further steps.

The validation/training data split was done using `sklearn.model_selection.train_test_split()` in lines `87-91`. 

I took advantage of the dataset provided from the Udacity team and ran my initial model being trained on just that data. Using this data, I preprocessed the images by resizing them to (32,32,3) for fast training of the DNN model. 

I also had attempted to use threshing techniques thinking this would give the DNN some features to understand from the beginning without having to learn them, but this also failed.

![alt text][image5]

This is an overview image of the vehicle, but the same threshing parameters were used on the for the images coming throuh the dataset. The threshing code can be seen in lines 10-71 in `model.py`. The results were initially promising with this technique, however, as the vehicle approached the first left curve after the bridge (see image below and above), the vehicle was unable to steer left from the dirt path. The vehicle continue to travel upon the dirt path, failing to stay on the paved road. 

![alt text][image3]

I attempted to fix this by recording just driving past this path multiple times, increasin the number of times the model would be trained on this curve, but I could not get promising results. I collected additional regular routes of the track as well as driving in the opposite direction to help the model generalize and add more right turns.

Seeing that I could not get the model to work with threshing, the final method used was a simple one. Using all the data I collected as well as the dataset provided, I trained the model with the raw RGB images of size (32,32,3) at 2 epochs using the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
