import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import pickle
import time

def cal_undistort(img, objpoints, imgpoints):
    start_cal = time.time()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    end_cal_start_undist = time.time()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    end_all = time.time()
    print("cal_time: {} undist_time: {}".format(end_cal_start_undist-start_cal, end_all-end_cal_start_undist))
    return undist

def get_cal_pts(images, nx, ny):
    # initialize object points and image points list
    objpts = []                         # 3D points in real-world space
    imgpts = []                         # 2D points in image

    # Prepare object points array
    objpt = np.zeros((nx*ny,3), np.float32)     
    objpt[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates

    for image in images:
        # Read image
        img = cv2.imread(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # find corners from chessboard function 
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If corners were detected, add image/object points to list
        if ret:
            imgpts.append(corners)
            objpts.append(objpt)
        else:
            print("Did not detect corners in: {}".format(image))         
    return objpts, imgpts

def main(pickle_file, from_pickle=False):
    if not from_pickle:
        # Import images to use for calibration
        images = glob.glob('camera_cal/calibration*')

        # Get object and image points
        object_points, image_points = get_cal_pts(images, 9, 6)

        # Get mtx and dist vectors
        img = cv2.imread('camera_cal/calibration1.jpg')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img.shape[1:], None, None)

        # Save points to a file to use for lane detection algorithm
        with open(pickle_file, 'wb') as f:
            pickle.dump({'object_points': object_points,
                         'image_points': image_points,
                         'mtx': mtx, 
                         'dist': dist}, f)
    else:
        dist_pickle = pickle.load(open(pickle_file, 'rb'))
        object_points = dist_pickle['object_points']
        image_points = dist_pickle['image_points']

    # Test calibration on an image
    img = cv2.imread('camera_cal/calibration1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist = cal_undistort(img, object_points, image_points)
    fig = plt.figure()
    img_ax = fig.add_subplot(1,2,1)
    img_ax.set_title('Original')
    img_ax.imshow(img)
    undist_ax = fig.add_subplot(1,2,2)
    undist_ax.set_title('Calibrated')
    undist_ax.imshow(undist)
    plt.show()

if __name__  == '__main__':
    main('cal_pts.p', False)