from styx_msgs.msg import TrafficLight
import cv2
from keras.models import load_model
from numpy import zeros, newaxis
import rospkg
import numpy as np
import tensorflow as tf 

class TLClassifier(object):
    def __init__(self):
	ros_root = rospkg.get_ros_root()
	r = rospkg.RosPack()
	path = r.get_path('tl_detector')
	print(path)
        self.model = load_model(path + '/model.h5') 

	self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img = cv2.resize(image, (400, 400)) 
        img = img.astype(float)
    	img = img / 255.0

    	img = img[newaxis,:,:,:]
	with self.graph.as_default():
            predicts = self.model.predict(img)
    	predicted_class = np.argmax(predicts, axis=1)

    	print('Predicted Class:' ,predicted_class[0])
    	lid = predicted_class[0]

        if(lid == 1):
           return TrafficLight.RED

        return TrafficLight.UNKNOWN
