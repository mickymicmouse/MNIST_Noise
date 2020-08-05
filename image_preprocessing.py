# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:46:41 2020

@author: seungjun
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


#----make black image----

def make_black(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# Create new blank 300x300 red image

black_image = make_black(84, 84)
plt.imshow(black_image)
plt.show()

path = r"E:\\MNIST\\all\\9\\4.png"
image = cv2.imread(path)
def image_replace(source, black_image, )

    #image random batch size 84,84
    
    height, width = image.shape[:2]
    c_height = np.random.randint(low=5, high = 84)
    c_width = c_height
    
    
    image = cv2.resize(image,(c_height,c_width))
    
    b_height = 84
    b_width = 84
    
    x = np.random.randint(low=0,high=(b_height-c_height))
    y = np.random.randint(low=0,high=(b_width - c_width))
    roi = black_image[x:x+c_height, y:y+c_width]
    
    roi_new = cv2.add(image, roi)
    #plt.imshow(roi_new)
    
    np.copyto(roi, roi_new)
    #plt.imshow(black_image)
    
    #image noise N(0,sigma) sigma sampling at N(30,30^2)
    
    s = 0
    while s<=0:
        s = np.random.normal(30, 30, 1)
        
    
    noise = np.random.normal(0, s, (84,84,1))
    noise = noise.reshape(84,84,1)
    result_image = black_image + noise
    #plt.imshow(result_image)
    #cv2.imshow("plt", result_image)
    
    return result_image
