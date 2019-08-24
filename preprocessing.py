from math import sqrt
import numpy as np
import cv2
import os
import sys
import time

start = time.time()


def distance_sort(array):
    '''Find distance of each point from (0, 0) and return sorted array as per the distance
    array = Contain the image pixel in an numpy array
    '''
    distances = []

    # Calculate Euclid Distance for sorting
    for val in array:
        dist = sqrt(abs(val[0][0] - 0) ** 2 + abs(val[0][1] - 0) ** 2)
        distances.append(dist)

    coords_dist = sorted(zip(distances, array), key=lambda dist: dist[0])

    return np.array([z for _, z in coords_dist], dtype=np.int32)

def Preprocess_OCT_Image(infolder, outfolder):
    '''
    Preprocess the images based on the system requirement
    :param infolder: The input directory of the OCT images file
    :param outfolder: THe output directoary where the processed images will be saved
    '''
    Images = os.listdir(infolder)
    for Image in Images:
        print(Image)
        infilename = os.path.join(infolder, Image)

        # Take image and convert to GrayScale
        bgrimg = cv2.imread(infilename, 1)
        bgrimg_copy = bgrimg.copy()
        grayimg = cv2.cvtColor(bgrimg_copy, cv2.COLOR_BGR2GRAY)

        # Threshold gray image to create a mask
        ret, mask = cv2.threshold(grayimg, 254, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Extract the region of interest (here the area apart from white region)
        roi_img = cv2.bitwise_and(grayimg, grayimg, mask=mask_inv)

        # Draw external contours to find boundary
        contourimage, contoursCV, hierarchy = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = max(contoursCV, key=cv2.contourArea)

        # Find approximate of contour with maximum area
        epsilon = 0.038 * cv2.arcLength(max_cnt, True)
        approx = cv2.approxPolyDP(max_cnt, epsilon, True)

        # Sort four approximate points to order them in (topleft, bottomleft, bottomright, topright)
        approx = distance_sort(approx)

        # Store and draw the corner points
        topLeft = approx[0][0]
        botRight = approx[len(approx) - 1][0]
        newapprox = np.core.records.fromarrays(approx[1:-1].transpose(), names='X, Y', formats='i8, i8')
        newapprox = np.sort(newapprox, order='X')
        botLeft = newapprox[0][0]

        newapprox = np.sort(newapprox, order='Y')
        indx = 0
        for i in range(1, len(newapprox[0]), 1):
            if newapprox[0][indx][1] != newapprox[0][i][1]:
                break
            indx += 1
        topRight = newapprox[0][indx]

        topLeft[0] += 30
        topLeft[1] += 30

        botRight[0] -= 30
        botRight[1] -= 30

        botLeft[0] += 30
        botLeft[1] -= 30

        topRight[0] -= 30
        topRight[1] += 50

        # Finding Perspective Transform of the given image to make to square in dimensions
        inPoints = np.float32([list(topLeft), list(botLeft), list(topRight), list(botRight)])
        outPoints = np.float32([[0, 0], [0, 512], [512, 0], [512, 512]])

        # Perspective Mask
        perspectiveMask = cv2.getPerspectiveTransform(inPoints, outPoints)

        # Warp Perspective
        grayimg = cv2.warpPerspective(grayimg, perspectiveMask, (512, 512))

        # Threshold and Sobel Y operator filtering
        denoised_img = cv2.fastNlMeansDenoising(grayimg, None, 15, 7, 21)
        ret, thresh = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sobely = cv2.Sobel(thresh, cv2.CV_64F, 0, 1, ksize=5)

        # Eroding Sobel Y image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gradient = cv2.morphologyEx(sobely, cv2.MORPH_GRADIENT, kernel)
        erosion = cv2.erode(gradient, kernel, iterations=1)

        # Removing extreme points
        cleankernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, cleankernel)
        clean = np.asarray(clean, np.uint8)

        # Finding contours and ROI

        _, contoursCV_wrap, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Finding corner points for diagonal of bounding rectangle
        maxPoint = [0, 0]
        minPoint = [0, 512]
        for cnts in contoursCV_wrap:
            pointmn = cnts[cnts[:, :, 1].argmin()][0]
            pointmx = cnts[cnts[:, :, 1].argmax()][0]

            if pointmn[1] <= minPoint[1]:
                minPoint = pointmn

            if pointmx[1] >= maxPoint[1]:
                maxPoint = pointmx

        minPoint[0] = 0
        maxPoint[0] = 511

        minPoint[1] = (minPoint[1] - 5) if (minPoint[1] - 5) > 0 else 0
        maxPoint[1] = (maxPoint[1] + 80) if (maxPoint[1] + 80) < 511 else 511

        # Region of interest cropping
        roi = denoised_img[minPoint[1]:maxPoint[1], minPoint[0]:maxPoint[0]]
        resized_roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)

        # Writing denoised ROI image to outfolder
        outfilename = os.path.join(outfolder, Image)
        cv2.imwrite(outfilename, resized_roi)

        print("Writing Image", Image, "to", outfolder)

#main body function
Path_in=sys.argv[1]
Path_out=sys.argv[2]
Preprocess_OCT_Image(Path_in,Path_out)

print(time.time()-start)
