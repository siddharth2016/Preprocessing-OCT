from xtract_features.region_props import *
from collections import OrderedDict
from math import copysign, log10
import pandas as pd
import numpy as np
import time
import sys
import cv2
import os

dictlist = []

def coef(x, y):
    x.astype(float)
    y.astype(float)
    n = len(x)
    a = []
    for i in range(n):
        a.append(y[i])

    for j in range(1, n):

        for i in range(n-1, j-1, -1):
            try:
                a[i] = float(a[i]-a[i-1])/float(x[i]-x[i-j])
            except ZeroDivisionError:
                a[i] = float(a[i]-a[i-1])
            try:
                a[i] = -1 * copysign(1.0, a[i]) * log10(abs(a[i]))
            except ValueError:
                a[i] = -1 * copysign(1.0, a[i])

    return np.array(a) # return an array of coefficient

def ExtractFeatures():
    for i in range(0, 1, 1):
        #Take Image input
        denoised_img = cv2.imread("13.jpeg", 1)
        copyimgs = denoised_img.copy()
        grayimg = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

        #Threshold and Sobel Y operator filtering
        ret,thresh = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        dilatekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        dilate = cv2.dilate(thresh, dilatekernel, iterations=1)

        retinakernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(39,39))
        retinaonly = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, retinakernel)

        #Erode Image and Clean Image
        erosionkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        erosion = cv2.erode(thresh, erosionkernel, iterations=1)

        cleankernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        clean = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, cleankernel)

        #Finding contours and ROI
        contourImage, contoursCV, hierarchy = cv2.findContours(clean,
                                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contourImage2, contoursCV2, hierarchy2 = cv2.findContours(retinaonly,
                                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sortedContours = sorted(contoursCV, key=cv2.contourArea, reverse=True)
        sortedContours2 = sorted(contoursCV2, key=cv2.contourArea, reverse=True)

        contour1y = sortedContours[0][sortedContours[0][:,:,1].argmax()]
        contour2y = [[0,0]] if len(sortedContours)==1 else sortedContours[1][sortedContours[1][:,:,1].argmax()]

        roi_contour_bottom = sortedContours[0]

        if contour1y[0][1]>=contour2y[0][1]:
            roi_contour_bottom = sortedContours[0]
        else:
            roi_contour_bottom = sortedContours[1]

        xarray = np.asarray(sortedContours2[0][:,:,0].flatten().tolist())
        #print(xarray)
        yarray = np.asarray(sortedContours2[0][:,:,1].flatten().tolist())
        #print(yarray)

        coeffs = coef(xarray, yarray)
        choroid_thickness = cv2.contourArea(roi_contour_bottom)
        retina_thickness = cv2.contourArea(sortedContours2[0])
        humoments = cv2.HuMoments(cv2.moments(thresh))
        eccentricity = region_props(grayimg).eccentricity()

        datadict = OrderedDict()
        datadict["choroid_thickness"] = choroid_thickness
        datadict["retina_thickness"] = retina_thickness
        datadict["eccentricity"] = eccentricity

        for i in range(0, len(humoments), 1):
            humoments[i] = -1 * copysign(1.0, humoments[i]) * log10(abs(humoments[i]))
            datadict["HuMoment_"+str(i+1)] = humoments[i][0]

        for i in range(1, 11, 1):
            datadict["coeff_"+str(i)] = coeffs[i]

        dictlist.append(datadict)

        cv2.drawContours(denoised_img, [roi_contour_bottom], -1, (0, 255, 0), 2)
        cv2.drawContours(copyimgs, [sortedContours2[0]], -1, (0, 255, 255), 2)
        
        cv2.imshow("1", denoised_img)
        cv2.imshow("2", copyimgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

ExtractFeatures()


