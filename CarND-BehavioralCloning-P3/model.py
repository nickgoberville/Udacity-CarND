#! /usr/bin/env python3
import pandas as pd
import numpy as np
import cv2
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda, Dropout, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import time

def colorspace_thresh(img, colorspace):
        ## V1

    if colorspace == 'BGR':
        low = [0,0,0]
        high = [255,255,255]
    elif colorspace == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        low = [0,0,125]
        high = [0,24,255]
    elif colorspace == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        low = [255,129,0]
        high = [255,255,0]
        
    ch0, ch1, ch2 = cv2.split(img)

        # V1
        
    binary0 = np.zeros_like(ch0)
    binary0[(ch0 >= low[0]) & (ch0 <= high[0])] = 1
        
    binary1 = np.zeros_like(ch1)
    binary1[(ch1 >= low[1]) & (ch1 <= high[1])] = 1
        
    binary2 = np.zeros_like(ch2)
    binary2[(ch2 >= low[2]) & (ch2 <= high[2])] = 1

    color_binary = np.dstack(( binary0, binary1, binary2))*255
        
    return color_binary

def final_canny(img):
    '''
    Using method from advanced lane finding
    '''
    BGR_binary = colorspace_thresh(img, 'BGR')
    HSV_binary = colorspace_thresh(img, 'HSV')
    HLS_binary = colorspace_thresh(img, 'HLS')
        
        # Gradient Threshing
    BGR_canny = cv2.Canny(BGR_binary, -67, 0)
    HSV_canny = cv2.Canny(HSV_binary, -67, 0)
    HLS_canny = cv2.Canny(HLS_binary, -67, 0)
        
        # Combing threshed images of the RGB, HSV, HLS spaces
    combo_canny = cv2.bitwise_or(BGR_canny, HSV_canny)
    combo_canny = cv2.bitwise_or(combo_canny, HLS_canny)
    return combo_canny

def preprocess(image):
    '''
    Prepocess image 
    '''
    image = cv2.resize(image, (320, 160))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sizex = image.shape[1]
    sizey = image.shape[0]
    #image = image[70:sizey-25, :]      ## Add cropping to NN model
    image = cv2.GaussianBlur(image, (3,3), 0)
    return image

'''def get_data(list_of_driving_logs):
    images = []
    steer_cmds = []
    for path in list_of_driving_logs:
        log = pd.read_csv(path+'driving_log.csv')
        for index in log.index:
            # Get image paths from csv
            left_img_path = path+'IMG/'+log.left.values[index].split('/')[-1]
            center_img_path = path+'IMG/'+log.center.values[index].split('/')[-1]
            right_img_path = path+'IMG/'+log.right.values[index].split('/')[-1]
            # preprocess images
            left_image = preprocess(cv2.imread(left_img_path))
            center_image = preprocess(cv2.imread(center_img_path))
            right_image = preprocess(cv2.imread(right_img_path))
            # get steering values for each image
            steer_adjustment = 0.2
            left_steer = log.steer.values[index] - steer_adjustment
            center_steer = log.steer.values[index]
            right_steer = log.steer.values[index] + steer_adjustment
            # append images and steering to lists for training
            images.append(left_image)
            images.append(center_image)
            images.append(right_image)
            steer_cmds.append(left_steer)
            steer_cmds.append(center_steer)
            steer_cmds.append(right_steer)
    return images, steer_cmds
'''
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
            left_steer = log.steer.values[index] + steer_adjustment
            center_steer = log.steer.values[index]
            right_steer = log.steer.values[index] - steer_adjustment
            # append images and steering to lists for training
            image_paths.append(left_img_path)
            image_paths.append(center_img_path)
            image_paths.append(right_img_path)
            steer_cmds.append(left_steer)
            steer_cmds.append(center_steer)
            steer_cmds.append(right_steer)
    return image_paths, steer_cmds

def get_correction_data(list_of_correction_logs):
    image_paths = []
    steer_cmds = []
    for path in list_of_correction_logs:
        log = pd.read_csv(path+'driving_log.csv')
        for index in log.index:
            side = path.split('/')[1].split('-')[0]
            if side == 'left':
                if log.steer.values[index] > 0:
                    # Get image paths from csv
                    left_img_path = path+'IMG/'+log.left.values[index].split('/')[-1]
                    center_img_path = path+'IMG/'+log.center.values[index].split('/')[-1]
                    right_img_path = path+'IMG/'+log.right.values[index].split('/')[-1]
                    # get steering values for each image
                    steer_adjustment = 0.2
                    left_steer = log.steer.values[index] + steer_adjustment
                    center_steer = log.steer.values[index]
                    right_steer = log.steer.values[index] - steer_adjustment
                    # append images and steering to lists for training
                    image_paths.append(left_img_path)
                    image_paths.append(center_img_path)
                    image_paths.append(right_img_path)
                    steer_cmds.append(left_steer)
                    steer_cmds.append(center_steer)
                    steer_cmds.append(right_steer)
            elif side == 'right':
                if log.steer.values[index] < 0:
                    # Get image paths from csv
                    left_img_path = path+'IMG/'+log.left.values[index].split('/')[-1]
                    center_img_path = path+'IMG/'+log.center.values[index].split('/')[-1]
                    right_img_path = path+'IMG/'+log.right.values[index].split('/')[-1]
                    # get steering values for each image
                    steer_adjustment = 0.2
                    left_steer = log.steer.values[index] - steer_adjustment
                    center_steer = log.steer.values[index]
                    right_steer = log.steer.values[index] + steer_adjustment
                    # append images and steering to lists for training
                    image_paths.append(left_img_path)
                    image_paths.append(center_img_path)
                    image_paths.append(right_img_path)
                    steer_cmds.append(left_steer)
                    steer_cmds.append(center_steer)
                    steer_cmds.append(right_steer)
    return image_paths, steer_cmds    

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
                #while n < len(X):
                #print('n: {} n_samples: {}'.format(n, num_samples))
                
                #while batch_n < batch_size:
                try:
                    #if np.random.choice(flip_opts):
                    #    print("n: {} flipping? {}".format(n, 'yes'))
                    #    img = cv2.flip(preprocess(cv2.imread(X[n])), 1)
                    #    ang = -1.0*y[n]
                    #else:
                    #    print("n: {} flipping? {}".format(n, 'no'))                        
                    img = preprocess(cv2.imread(X[n]))
                    ang = y[n]                        
                    X_batch.append(img)
                    y_batch.append(ang)
                except:
                    #batch_n -= 1
                    #n -= 1
                    continue
                batch_n += 1
                #n += 1
                    #print(n)
            yield np.array(X_batch), np.array(y_batch)




def main():
    batch_size=128
    # set lists of directories to use data from
    list_of_driving_logs = ['data/data/']#['data/updated-center-normal/', 'data/center-normal/', 'data/center-reverse/', 'data/center-normal2/']#]
    # set lists of directories to use for data correction
    #list_of_correction_logs = ['data/left-normal/', 'data/right-normal/']
    # Get list of image paths and steering commands
    image_paths, steer_cmds = get_data(list_of_driving_logs)
    #correction_img_paths, correction_steer_cmds = get_correction_data(list_of_correction_logs)
    # appending correction data to image_paths and steer_cmds
    #image_paths.append(correction_img_paths)
    #steer_cmds.append(correction_steer_cmds)
    # Split and shuffle data
    X_test, X_train, y_test, y_train = train_test_split(image_paths, steer_cmds, train_size=0.1, shuffle=True)
    X_train = X_train[0:-(len(X_train)%batch_size)]
    X_test = X_test[0:-(len(X_test)%batch_size)]
    y_train = y_train[0:-(len(y_train)%batch_size)]
    y_test = y_test[0:-(len(y_test)%batch_size)]
    #print("Test...X: {} y: {}\nTrain...X: {} y: {}".format(len(X_test)//batch_size, len(y_test), len(X_train)//batch_size, len(y_train)))
    time.sleep(5)
    # Get image to use for input_shape of DNN
    ex_img = preprocess(cv2.imread(X_train[950]))
    ex_img_flip = cv2.flip(ex_img, 1)
    cv2.imwrite('ex_img.png', cv2.cvtColor(ex_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite('flipped.png', cv2.cvtColor(ex_img_flip, cv2.COLOR_BGR2RGB))
    print('ex_img steer: {}'.format(y_train[950]))
    #cv2.waitKey(5000)
    # Define train and test generator variables
    train_generator = generator((X_train, y_train), batch_size)
    test_generator = generator((X_test, y_test), batch_size)

    # DNN Architecture
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=ex_img.shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    model.add(Flatten())
    model.add(Dropout(0.25))    
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Compile using adam optimizer and fit to train/validation data
    model.compile('adam', loss='mse')
    #history_object = model.fit(images, steer_cmds, verbose=1, validation_split=0.2, epochs=10)
    history_object = model.fit_generator(train_generator, steps_per_epoch=math.floor(len(X_train)/batch_size), validation_data=test_generator, validation_steps=math.floor(len(X_test)/batch_size), epochs=5, verbose=1)

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