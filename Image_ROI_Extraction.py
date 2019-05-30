import numpy as np
import time
import cv2
import os

start_time = time.time()
infolder = "warpped_dataset/train/AMD"
outfolder = "retina_dataset/train/AMD"
Images = os.listdir(infolder)

for Image in Images:
    print(Image)
    infilename = os.path.join(infolder, Image)

    #Take Image input
    bgrimg = cv2.imread(infilename, 1)
    grayimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)

    #Threshold and Sobel Y operator filtering
    denoised_img = cv2.fastNlMeansDenoising(grayimg, None, 15, 7, 21)
    ret,thresh = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    sobely = cv2.Sobel(thresh,cv2.CV_64F,0,1,ksize=5)

    #Eroding Sobel Y image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    gradient = cv2.morphologyEx(sobely, cv2.MORPH_GRADIENT, kernel)
    erosion = cv2.erode(gradient, kernel, iterations=1)

    #Removing extreme points
    cleankernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, cleankernel)
    clean = np.asarray(clean, np.uint8)

    #Finding contours and ROI
    contourImage, contoursCV, hierarchy = cv2.findContours(clean,
                                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Finding corner points for diagonal of bounding rectangle
    maxPoint = [0, 0]
    minPoint = [0, 512]
    for cnts in contoursCV:
        pointmn = cnts[cnts[:,:,1].argmin()][0]
        pointmx = cnts[cnts[:,:,1].argmax()][0]

        if pointmn[1]<=minPoint[1]:
            minPoint = pointmn

        if pointmx[1]>=maxPoint[1]:
            maxPoint = pointmx
        
    minPoint[0] = 0
    maxPoint[0] = 511

    minPoint[1] = (minPoint[1]-5) if (minPoint[1] - 5)>0 else 0
    maxPoint[1] = (maxPoint[1]+80) if (maxPoint[1] + 80)<511 else 511

    #Region of interest cropping
    roi = denoised_img[minPoint[1]:maxPoint[1], minPoint[0]:maxPoint[0]]
    resized_roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)

    #Writing denoised ROI image to outfolder
    outfilename = os.path.join(outfolder, Image)
    cv2.imwrite(outfilename, resized_roi)

    print("Writing Image", Image, "to", outfolder)

print("Time Elapsed ---", (time.time()-start_time))
