#Invisible Cloak project using OpenCV.

import cv2 #Library
import time
import numpy as np

#To access the camera,and set the capture object as cap 
cap = cv2.VideoCapture(0)  

# Store a single frame as background 
_, background = cap.read() #Captures frames from webcam
time.sleep(2) #Adjusting camera auto exposure
_, background = cap.read()

#Define all the kernels size.
#A kernel is a small matrix used for blurring, sharpening,edge detection,and more.  
open_kernel = np.ones((5,5),np.uint8) #Create a 5*5 8bit integer matrix
close_kernel = np.ones((7,7),np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

# Function for remove noise from mask.
# Image noise is random variation of brightness or color information in the images captured. 
# A mask allows us to focus only on the portions of the image that interests us.
def filter_mask(mask):

    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)#Removes unnecessary black noise from the white region in the mask

    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel) #Removes unnecessary white noise from the black region

    dilation = cv2.dilate(open_mask, dilation_kernel, iterations= 1)#Increases white region in the image

    return dilation

while cap.isOpened():
    ret, frame = cap.read()  # Capture every frame
    # convert to hsv colorspace 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound are the boundaries for Green color 
    lower_bound = np.array([50, 80, 50])     
    upper_bound = np.array([90, 255, 255])
    

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Filter mask
    mask = filter_mask(mask)

    # Apply the mask to take only those region from the saved background 
    # where our cloak is present in the current frame
    cloak = cv2.bitwise_and(background, background, mask=mask)#Applies mask on frame in the region where mask is true(mean white).

    # create inverse mask 
    inverse_mask = cv2.bitwise_not(mask) #Inverse the mask pixel value

    # Apply the inverse mask to take those region of the current frame where cloak is not present 
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine cloak region and current_background region to get final frame 
    combined = cv2.add(cloak, current_background) #Adds two frames and returns a single frame.

    cv2.imshow("Final output", combined)


    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

