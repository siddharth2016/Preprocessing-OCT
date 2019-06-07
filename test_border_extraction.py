import cv2
import os
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

#Take Image input
bgrimg = cv2.imread("12.jpeg", 1)
grayimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)

#Threshold and Sobel Y operator filtering
denoised_img = cv2.fastNlMeansDenoising(grayimg, None, 15, 7, 21)
ret,thresh = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
sobely = cv2.Sobel(thresh,cv2.CV_64F,0,1,ksize=5)
#sobelx = cv2.Sobel(thresh, cv2.CV_64F, 1, 0, ksize=5)

#Eroding Sobel Y image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
gradient = cv2.morphologyEx(sobely, cv2.MORPH_GRADIENT, kernel)
erosion = cv2.erode(gradient, kernel, iterations=1)

#Removing extreme points
cleankernel = np.ones((3,3), np.uint8)
clean = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, cleankernel)
clean = np.asarray(clean, np.uint8)

#Finding contours and ROI
image1, contoursCV, hierarchy = cv2.findContours(clean,
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Finding corner points for diagonal of bounding rectangle
maxPoint = [0, 0]
minPoint = [0, 300]
for cnts in contoursCV:
    pointmn = cnts[cnts[:,:,1].argmin()][0]
    pointmx = cnts[cnts[:,:,1].argmax()][0]

    if pointmn[1]<=minPoint[1]:
        minPoint = pointmn

    if pointmx[1]>=maxPoint[1]:
        maxPoint = pointmx
    
#print(maxPoint, minPoint)
minPoint[0] = 0
maxPoint[0] = 511

minPoint[1] = (minPoint[1]-10) if (minPoint[1] - 10)>0 else 0
maxPoint[1] = (maxPoint[1]+90) if (maxPoint[1] + 90)<511 else 511

#print("updated", maxPoint, minPoint)

roi = denoised_img[minPoint[1]:maxPoint[1], minPoint[0]:maxPoint[0]]
resized_roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_CUBIC)

#contourimg = cv2.drawContours(colorimg, [contoursCV[len(contoursCV)-1]], -1, (0, 255, 255), 2)
#contourimg = cv2.drawContours(colorimg, [contoursCV[0]], -1, (0, 255, 0), 2)

cv2.imshow("clean", clean)
cv2.imshow("denoised", denoised_img)
cv2.imshow("sobel y", sobely)
cv2.imshow("thresh", thresh)
cv2.imshow("gradient", gradient)
cv2.imshow("roi", resized_roi)
#cv2.imshow("sobel x", sobelx)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("13.jpeg", resized_roi)
# Matplotlib Method
contoursMAT = np.asarray(measure.find_contours(clean, 0.8))

fig, ax = plt.subplots()
ax.imshow(bgrimg, interpolation='none', cmap=plt.cm.gray)

for n, contour in enumerate(contoursMAT):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

