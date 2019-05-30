import cv2
import os
import numpy as np

infolder = "retina_dataset/train/AMD"
Images = os.listdir(infolder)

for image in Images:
    print(image)
    filename = os.path.join(infolder, image)

    #Take Image input
    denoised_img = cv2.imread(filename,  1)
    grayimg = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

    #Threshold and Sobel Y operator filtering
    ret,thresh = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #Clean Most Extreme Points
    erosionkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(thresh, erosionkernel, iterations=1)

    cleankernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    clean = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, cleankernel)

    #Finding contours and ROI
    contourImage, contoursCV, hierarchy = cv2.findContours(clean,
                                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sortedContours = sorted(contoursCV, key=cv2.contourArea, reverse=True)

    contour1y = sortedContours[0][sortedContours[0][:,:,1].argmax()]
    contour2y = [[0,0]] if len(sortedContours)==1 else sortedContours[1][sortedContours[1][:,:,1].argmax()]

    #Findind ROI Contour and its Area Feature
    roi_contour = sortedContours[0]

    if contour1y[0][1]>=contour2y[0][1]:
        roi_contour = sortedContours[0]
    else:
        roi_contour = sortedContours[1]

    
    print("Area Feature of Choroid", cv2.contourArea(roi_contour))
    

