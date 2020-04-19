#! /usr/bin/env python3
import pandas as pd
import numpy as np
import cv2
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda, Dropout, Cropping2D
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import time

def preprocess(image):
    '''
    Prepocess image 
    '''
    if type(None) == type(image):
      return image
    sizey = image.shape[0]    
    image = cv2.cvtColor(cv2.resize(image[70:sizey-25, :], (32,32)), cv2.COLOR_BGR2RGB)
    return image

def get_data(list_of_driving_logs):
    image_paths = []
    steer_cmds = []
    for path in list_of_driving_logs:
        log = pd.read_csv(path+'driving_log.csv')
        print("Getting log from {}.".format(path))
        for index in log.index:

            # Get image paths from csv
            left_img_path = path+'IMG/'+log.left.values[index].split('/')[-1]
            center_img_path = path+'IMG/'+log.center.values[index].split('/')[-1]
            right_img_path = path+'IMG/'+log.right.values[index].split('/')[-1]

            # get steering values for each image
            steer_adjustment = 0.2
            left_steer = log.steering.values[index] + steer_adjustment
            center_steer = log.steering.values[index]
            right_steer = log.steering.values[index] - steer_adjustment

            # append images and steering to lists for training
            image_paths.append(left_img_path)
            image_paths.append(center_img_path)
            image_paths.append(right_img_path)
            steer_cmds.append(left_steer)
            steer_cmds.append(center_steer)
            steer_cmds.append(right_steer)
    return image_paths, steer_cmds   

#generator definition
def generator(samples, batch_size): 
    n = 0
    num_samples = len(samples[0])
    X = samples[0]
    y = samples[1]
    flip_opts = np.array([True, False])
    while True:
        for offset in range(0, num_samples, batch_size):
            X_batch = []
            y_batch = []
            batch_n = 0
            for n in range(offset, offset+batch_size):
                #add random flipping
                if np.random.choice(flip_opts):
                    img = cv2.flip(preprocess(cv2.imread(X[n])), 1)
                    ang = -1.0*y[n]
                    X_batch.append(img)
                    y_batch.append(ang)                        
                img = preprocess(cv2.imread(X[n]))
                ang = y[n]                        
                X_batch.append(img)
                y_batch.append(ang)
                batch_n += 1
            yield np.array(X_batch), np.array(y_batch)


def main():
    batch_size=128

    # set lists of directories to use data from
    list_of_driving_logs = ['data/data/']
    image_paths, steer_cmds = get_data(list_of_driving_logs)

    # Split and shuffle data (testing size 0f 20%)
    image_paths, steer_cmds = shuffle(image_paths, steer_cmds)
    X_train, X_test, y_train, y_test = train_test_split(image_paths, steer_cmds, test_size=0.2, shuffle=False)
    #shuffle the data again
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    
    #remove excess data for each distribution among batches
    X_train = X_train[0:-(len(X_train)%batch_size)]
    X_test = X_test[0:-(len(X_test)%batch_size)]
    y_train = y_train[0:-(len(y_train)%batch_size)]
    y_test = y_test[0:-(len(y_test)%batch_size)]


    # Get image to use for input_shape of DNN & save example of image and a flipped image
    ex_img = preprocess(cv2.imread(X_train[950]))
    ex_img_flip = cv2.flip(ex_img, 1)
    cv2.imwrite('ex_img.png', cv2.cvtColor(ex_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite('flipped.png', cv2.cvtColor(ex_img_flip, cv2.COLOR_BGR2RGB))

    # Define train and test generator variables
    train_generator = generator((X_train, y_train), batch_size)
    test_generator = generator((X_test, y_test), batch_size)

    # DNN V2
    from keras.models import Sequential
    from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda
    from keras.callbacks import EarlyStopping

    #creating model to be trained
    model = Sequential()
    model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(32,32,3) ))
    model.add(Convolution2D(15, 3, 3, subsample=(2, 2), activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(1))

    # DNN Architecture -- NOT USED
    #model = Sequential()
    #model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=ex_img.shape))
    #model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    #model.add(Conv2D(24,5,5, subsample=(2,2), activation='elu'))
    #model.add(Conv2D(36,5,5, subsample=(2,2), activation='elu'))
    #model.add(Conv2D(48,5,5, subsample=(2,2), activation='elu'))
    #model.add(Conv2D(64,3,3, activation='elu'))
    #model.add(Conv2D(64,3,3, activation='elu'))
    #model.add(Dropout(0.8))    
    #model.add(Flatten())
    #model.add(Dense(100))
    #model.add(Dropout(0.6))
    #model.add(Dense(50))
    #model.add(Dense(10))
    #model.add(Dense(1))

    # Compile using adam optimizer and fit to train/validation data
    model.compile('adam', loss='mse', metrics=['accuracy'])
    
    #add early stopping callback to stop once overfitting occurs
    es = EarlyStopping(verbose=1)
    history_object = model.fit_generator(train_generator, steps_per_epoch=math.floor(len(X_train)/batch_size), validation_data=test_generator, validation_steps=math.floor(len(X_test)/batch_size), epochs=15, verbose=1, callbacks=[es])

    # Model results visualization
    fig = plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('model.png')
    plt.show()

    # Save the model
    model.save('model.h5')

if __name__ == '__main__':
    main()
