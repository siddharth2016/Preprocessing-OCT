# Pre-Processing Images


import os
import numpy as np
import cv2

def imagePreProcessing(img_path, scaled_path="processed_oct_normal", height=256, width=256):

    images = os.listdir(img_path)
    os.mkdir(scaled_path)

    for image in images:
        print("Processing Image", image)
        
        img = cv2.imread(os.path.join(img_path,image), 1)
        
        img = cv2.resize(img, (height, width))      #Image Scaling to (256, 256)

        img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)        #Image DeNoising

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      #BGR image to HSV image

        img = cv2.GaussianBlur(img, (7,7), 0)       #Gaussian Blurring image
        
        cv2.imwrite(os.path.join(scaled_path, image), img)


#Calling imagePreProcessing
imagePreProcessing("octnormal")
