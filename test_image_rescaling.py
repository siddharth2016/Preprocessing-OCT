import cv2
import numpy as np
from math import sqrt

#Take image and convert to GrayScale
bgrimg = cv2.imread("9.jpeg", 1)
bgrimg_copy = bgrimg.copy()
grayimg = cv2.cvtColor(bgrimg_copy, cv2.COLOR_BGR2GRAY)

#Threshold gray image to create a mask
ret, mask = cv2.threshold(grayimg, 254, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#Extract the region of interest (here the area apart from white region)
roi_img = cv2.bitwise_and(grayimg, grayimg, mask=mask_inv)

#Draw external contours to find boundary
contourimage, contoursCV, hierarchy = cv2.findContours(roi_img,
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_cnt = max(contoursCV, key=cv2.contourArea)

#Find approximate of contour with maximum area
epsilon = 0.038 * cv2.arcLength(max_cnt, True)
approx = cv2.approxPolyDP(max_cnt, epsilon, True)
#print(approx, type(approx), approx.shape, approx.dtype, approx.size)

