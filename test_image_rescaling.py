import cv2
import numpy as np
from math import sqrt

#Find distance of each point from (0, 0) and return sorted array as per the distance
def distance_sort(array):
    distances = []

    #Calculate Manhattan Distance for sorting
    for val in array:
        dist = sqrt(abs(val[0][0] - 0)**2 + abs(val[0][1] - 0)**2)
        distances.append(dist)

    coords_dist = sorted(zip(distances, array), key=lambda dist: dist[0])

    return np.array([z for _,z in coords_dist], dtype=np.int32)


#Take image and convert to GrayScale
bgrimg = cv2.imread("1.jpeg", 1)
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

#Sort four approximate points to order them in (topleft, bottomleft, bottomright, topright)
approx = distance_sort(approx)
print(approx, type(approx), approx.shape, approx.dtype, approx.size, len(approx))

#Store and draw the corner points
topLeft = approx[0][0]
botRight = approx[len(approx)-1][0]

newapprox = np.core.records.fromarrays(approx[1:-1].transpose(), names='X, Y',
                                             formats = 'i8, i8')
newapprox = np.sort(newapprox, order='X')
botLeft = newapprox[0][0]

print(newapprox, len(newapprox))

newapprox = np.sort(newapprox, order='Y')
indx = 0
for i in range(1, len(newapprox[0]), 1):
    if newapprox[0][indx][1]!=newapprox[0][i][1]:
        break
    indx+=1 
topRight = newapprox[0][indx]


topLeft[0]+=30
topLeft[1]+=30

botRight[0]-=30
botRight[1]-=30

botLeft[0]+=30
botLeft[1]-=30

topRight[0]-=30
topRight[1]+=50


print(newapprox, len(newapprox))
print(topLeft, botLeft, botRight, topRight)
#print(approx[1:-1])

cv2.drawContours(bgrimg_copy, [contoursCV[len(contoursCV)-1]], -1, (0, 0, 255), 2)

cv2.circle(bgrimg_copy, tuple(topLeft), 8, (0, 0, 255), -1)
cv2.circle(bgrimg_copy, tuple(botLeft), 8, (0, 255, 0), -1)
cv2.circle(bgrimg_copy, tuple(botRight), 8, (255, 0, 0), -1)
cv2.circle(bgrimg_copy, tuple(topRight), 8, (255, 255, 0), -1)

#Finding Perspective Transform of the given image to make to square in dimensions
inPoints = np.float32([list(topLeft), list(botLeft), list(topRight), list(botRight)])
outPoints = np.float32([[0,0], [0, 512], [512, 0], [512, 512]])

#Perspective Mask
perspectiveMask = cv2.getPerspectiveTransform(inPoints, outPoints)

#Warp Perspective
warppedImage = cv2.warpPerspective(bgrimg, perspectiveMask, (512, 512))

#Show Result Images
cv2.imshow("gray", grayimg)
cv2.imshow("bgrimg", bgrimg)
cv2.imshow("bgrimg_copy", bgrimg_copy)
cv2.imshow("warpimg", warppedImage)
cv2.imshow("mask_inv", mask_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(perspectiveMask)

cv2.imwrite("12.jpeg", warppedImage)

#cv2.imwrite("10.jpeg", warppedImage)
