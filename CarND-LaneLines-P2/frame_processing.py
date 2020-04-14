import cv2
import numpy as np
import math
import json
from camera_cal import cal_undistort
import pickle

global count_detect, left_fit, right_fit, mtx, dist
count_detect=0
pickle_file = 'cal_pts.p'
dist_pickle = pickle.load(open(pickle_file, 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

class processFrame:
    def __init__(self, frame, Params):
        '''
        Input must be frame and dict containing all parameters (showing defaults):
            cannyThresh1=0, cannyThresh2=0, gaussksize = (5,5), gausssigmaX = 0,
            adaptivemaxVal = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C,
            adaptiveType = cv2.THRESH_BINARY, adaptiveblockSize = 0, adaptiveC=0, 
            AOIhorizonHeight=100, AOIlinewidth=30, AOIbaseOffset=0, morphopenSize=5, morphcloseSize=5
        '''
        # Get frame parameters

        self.Params = Params
        self.frame = cv2.undistort(frame, mtx, dist, None, mtx)

        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]

        # Parameters for gauss blur
        self.gaussksize = (self.getParam('gaussksize')[0], self.getParam('gaussksize')[0])
        self.gausssigmaX = self.getParam('gausssigmaX')[0]

        # Parameters for canny
        self.cannyThresh1 = self.getParam('cannyThresh1')[0]
        self.cannyThresh2 = self.getParam('cannyThresh2')[0]
        self.apertureSize = self.getParam('apertureSize')[0]
        self.L2gradient = False#self.getParam('L2gradient')

        # Parameters for adaptiveThreshold
        self.adaptivemaxVal = self.getParam('adaptivemaxVal')[0]
        self.adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
        self.adaptiveType = cv2.THRESH_BINARY
        self.adaptiveblockSize = self.getParam('adaptiveblockSize')[0]
        self.adaptiveC = self.getParam('adaptiveC')[0]

        # Parameters for region of interest
        self.AOIhorizonHeight = self.getParam('AOIhorizonHeight')[0]
        self.AOIlinewidth = self.getParam('AOIlinewidth')[0]
        self.AOIbaseOffset = self.getParam('AOIbaseOffset')[0]
        self.AOIbasewidth = self.getParam('AOIbasewidth')[0]
        self.AOIxshift = -20

        # Parameters for morphing
        self.morphopenSize = self.getParam('morphopenSize')[0]
        self.morphcloseSize = self.getParam('morphcloseSize')[0]

        # Parameters for perspective image
        self.perspectPt_a = self.getParam('perspectPt_a')
        self.perspectPt_b = self.getParam('perspectPt_b')
        self.perspectPt_c = self.getParam('perspectPt_c')
        self.perspectPt_d = self.getParam('perspectPt_d')

        # Hough transformation Parameters
        self.houghResolution = self.getParam('houghResolution')[0]
        self.houghPiRes = np.pi / self.getParam('houghPiRes')[0]
        self.houghThresh = self.getParam('houghThresh')[0]
        self.houghLength = self.getParam('houghLength')[0]
        self.houghGap = self.getParam('houghGap')[0]

        # Lane color parameters
        self.leftlaneColor = self.getParam('leftlaneColor')
        self.rightlaneColor = self.getParam('rightlaneColor')
        self.laneThickness = self.getParam('laneThickness')[0]

        # Parameters for color space threshing
        self.HSV_low = self.getParam('HSV_low')
        self.HSV_high = self.getParam('HSV_high')
        self.HLS_low = self.getParam('HLS_low')
        self.HLS_high = self.getParam('HLS_high')
        self.BGR_low = self.getParam('BGR_low')
        self.BGR_high = self.getParam('BGR_high')

    def getParam(self, param):
        try:
            return self.Params[param]
        except KeyError:
            print("KeyError: {} is not in dict Param".format(param))

        try:
            self.count+=1
        except:
            self.count=0

    def perspective_transform(self, opened_img, ratio=2, use_AOI=False):
        # Either hard-coded perspectPt values or AOI points based on AOI parameters defines in __init__
        if use_AOI:
            perspective_pts = np.float32([
            [self.width//2-self.AOIbasewidth+self.AOIxshift, self.height-self.AOIbaseOffset],    
            [self.width//2-self.AOIlinewidth+self.AOIxshift, self.height-self.AOIhorizonHeight],
            [self.width//2+self.AOIlinewidth+self.AOIxshift, self.height-self.AOIhorizonHeight],
            [self.width//2+self.AOIbasewidth+self.AOIxshift, self.height-self.AOIbaseOffset]])  # carla
            
            perspective_array = np.array([
            [self.width//2-self.AOIbasewidth+self.AOIxshift, self.height-self.AOIbaseOffset],    
            [self.width//2-self.AOIlinewidth+self.AOIxshift, self.height-self.AOIhorizonHeight],
            [self.width//2+self.AOIlinewidth+self.AOIxshift, self.height-self.AOIhorizonHeight],
            [self.width//2+self.AOIbasewidth+self.AOIxshift, self.height-self.AOIbaseOffset],], np.int32)  # carla
        else:
            perspective_pts = np.float32([self.perspectPt_a,
                                      self.perspectPt_b,
                                      self.perspectPt_c,
                                      self.perspectPt_d])
            perspective_array = np.array([self.perspectPt_a,
                                      self.perspectPt_b,
                                      self.perspectPt_c,
                                      self.perspectPt_d], np.int32)

        # apply perspective transformation
        size_x = self.width
        size_y = self.width*ratio
        pts_2 = np.float32(
            [[0, size_y], [0, 0], [size_x, 0], [size_x, size_y]])
        mat = cv2.getPerspectiveTransform(perspective_pts, pts_2)
        invMat = cv2.getPerspectiveTransform(pts_2, perspective_pts)
        transformed_open = cv2.warpPerspective(opened_img, mat, (size_x, size_y))
        return transformed_open, mat, invMat, perspective_array

        color = self.frame.copy()  # save for perspective editor later

        # Compute canny image with thresholding
        canny_img, opened = self.canny(do_thresh=True)
        
        # Get image mask and masking points
        AOI_mask, AOI_left_mask, AOI_right_mask, AOI_pts = self.get_AOI_mask()
        # Get masked canny image
        masked_canny_img = self.get_masked(canny_img, AOI_mask)
        left_canny_img = self.get_masked(masked_canny_img, AOI_left_mask)
        right_canny_img = self.get_masked(masked_canny_img, AOI_right_mask)
        
        # Split masked canny image into left and right parts
        #left_canny_img, right_canny_image = self.split_image(canny_img)

        # Calculate left and right lines
        linesP_left = self.get_Hough_lines(left_canny_img)
        linesP_right = self.get_Hough_lines(right_canny_img)

        color = self.draw_lines(linesP_left, color, self.leftlaneColor)
        color = self.draw_lines(linesP_right, color, self.rightlaneColor)

        # IDK
        #canvas = self.get_black()

        # draw horizon mask on grayscale image for reference
        cv2.polylines(color, [AOI_pts], True, (255, 255, 255), 1)
        center_line = np.array([
            [int(self.width/2), self.AOIhorizonHeight],
            [int(self.width/2), self.height-self.AOIbaseOffset],], np.int32)
        cv2.polylines(color, [center_line], True, (255, 0, 255), 1)

        transformed_open, M, IM, perspective_pts = self.perspective_transform(opened)
        cv2.polylines(color, [perspective_pts], True, (255, 255, 0), 1)
        return color, transformed_open, canny_img, opened

    def colorspace_thresh(self, colorspace):
        ## V1
        
        img = self.frame.copy()
        if colorspace == 'BGR':
            low = self.BGR_low
            high = self.BGR_high
        elif colorspace == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            low = self.HLS_low
            high = self.HLS_high
        elif colorspace == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            low = self.HSV_low
            high = self.HSV_high
        
        ch0, ch1, ch2 = cv2.split(img)

        # V1
        
        binary0 = np.zeros_like(ch0)
        binary0[(ch0 >= low[0]) & (ch0 <= high[0])] = 1
        
        binary1 = np.zeros_like(ch1)
        binary1[(ch1 >= low[1]) & (ch1 <= high[1])] = 1
        
        binary2 = np.zeros_like(ch2)
        binary2[(ch2 >= low[2]) & (ch2 <= high[2])] = 1
        '''

        # V2
        _, binary0 = cv2.threshold(ch0, low[0], high[0], cv2.THRESH_BINARY)
        _, binary1 = cv2.threshold(ch1, low[1], high[1], cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(ch2, low[2], high[2], cv2.THRESH_BINARY)
        color_binary = cv2.merge((binary0, binary1, binary2))
        # Stack each channel
        '''
        color_binary = np.dstack(( binary0, binary1, binary2))*255
        
        return color_binary

    def measure_curvature_pixels(self, ploty):
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        ym_per_pix = 30/720
        xm_per_pix = 3.7/700
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        return left_curverad, right_curverad

    def measure_center_dist(self, leftx, rightx):
        xm_per_pix = 3.7/(470*0.6)
        dist_from_center = (self.width/2 - (np.median(leftx)+np.median(rightx))/2)*xm_per_pix
        return dist_from_center

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, binary_warped):
        global left_fit, right_fit
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        global left_fit, right_fit
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        return left_fitx, right_fitx, ploty, left_fit, right_fit

    def search_around_poly(self, binary_warped):
        global left_fit, right_fit
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 50

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty, left_fit, right_fit = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

        return result, left_fitx, right_fitx, left_fit, right_fit

    def detect_lanesV2(self, lanes=True):
        # Defining global variables
        global count_detect, left_fit, right_fit
        
        # Copy of frame
        img = self.frame.copy()

        # Color Threshing
        BGR_binary = self.colorspace_thresh('BGR')
        HSV_binary = self.colorspace_thresh('HSV')
        HLS_binary = self.colorspace_thresh('HLS')
        
        # Gradient Threshing
        BGR_canny = cv2.Canny(BGR_binary, self.cannyThresh1, self.cannyThresh2, apertureSize=self.apertureSize)
        HSV_canny = cv2.Canny(HSV_binary, self.cannyThresh1, self.cannyThresh2, apertureSize=self.apertureSize)
        HLS_canny = cv2.Canny(HLS_binary, self.cannyThresh1, self.cannyThresh2, apertureSize=self.apertureSize)
        
        # Combing threshed images of the RGB, HSV, HLS spaces
        combo_canny = cv2.bitwise_or(BGR_canny, HSV_canny)
        combo_canny = cv2.bitwise_or(combo_canny, HLS_canny)

        # Birds-eye perspective warping
        perspective_img, mat, invMat, perspective_pts = self.perspective_transform(combo_canny, ratio=1, use_AOI=True)       

        # If count_detect==0, search for lane_pixels with larger region
        # else, search around previously detected polynomial 
        if count_detect == 0:
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(perspective_img)
            combine, left_fitx, right_fitx, ploty, left_fit, right_fit = self.fit_polynomial(perspective_img)
        else:
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(perspective_img)
            combine, left_fitx, right_fitx, left_fit, right_fit = self.search_around_poly(perspective_img)
            ploty = np.linspace(0, perspective_img.shape[0]-1, perspective_img.shape[0] )
            
            # If large amount of error, reiterate the search using larger region
            if left_fitx[-1] >=300 or right_fitx[-1] <= 300:
                count_detect=-1
        count_detect+=1

        # Radius of curvature calculation
        leftcurve, rightcurve = self.measure_curvature_pixels(ploty)
        rofCurve = (leftcurve+rightcurve)/2
        
        # Distance from center calculation
        dist_from_center = self.measure_center_dist(leftx, rightx)
        
        ########## FOR VISUALIZATION ##########
        ### Calculating points for visualization
        # right & left points to plot on perspective transform image
        right_pts = np.array([np.asarray([right_fitx, ploty], dtype='float32').T])
        left_pts = np.array([np.asarray([left_fitx, ploty], dtype='float32').T])

        # Using inverse matrix to get polynomials to plot back onto origional image
        orig_pts_r = cv2.perspectiveTransform(right_pts, invMat)
        orig_pts_l = cv2.perspectiveTransform(left_pts, invMat)
        
        ### Adding all additional lines and annotations to images
        # radius of curvature and center text
        cv2.putText(img, "Radius of Curvature: {} m".format(round(rofCurve, 2)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
        cv2.putText(img, "Distance from center: {} m".format(round(dist_from_center, 2)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
        
        # left and right lanes on original image
        cv2.polylines(img, np.int32(orig_pts_r), False, (255,0,0), 2)
        cv2.polylines(img, np.int32(orig_pts_l), False, (0,0,255), 2)
        
        # green colored region between detected lines
        pts = np.hstack((orig_pts_l, np.fliplr(orig_pts_r)))
        cv2.fillPoly(img, np.int_([pts]), (0,100, 0))        
        
        # Plot polylines on perspective transform image
        cv2.polylines(combine, np.int32(right_pts), False, (255,0,255), 2)
        cv2.polylines(combine, np.int32(left_pts), False, (255,0,255), 2)        

        # UNCOMMET BELOW TO SHOW PERSPECTIVE POINTS ON ORIGINAL IMAGE
        #cv2.polylines(img, [perspective_pts], False, (0,0,255), 2)

        return BGR_binary, HSV_binary, HLS_binary, BGR_canny, HSV_canny, HLS_canny, combo_canny, img, combine, perspective_img

def save_values():
    with open("parameters.json", "w") as f:
        f.write(json.dumps(PARAM))

    print("saved values!")

def load_saved_values():
    global PARAM
    try:
        with open("parameters.json", "r") as f:
            data = json.loads(f.read())
            print(data)
            for key, value in data.items():
                PARAM[key] = value
    except FileNotFoundError:
        print("(didn't find parameters.json)")
