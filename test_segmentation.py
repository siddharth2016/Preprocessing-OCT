import cv2
import numpy as np
from scipy import ndimage

colorimg = cv2.imread("1.jpeg", 1)
grayimg = cv2.cvtColor(colorimg, cv2.COLOR_BGR2GRAY)

#img = cv2.resize(img, (256, 256))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#img = cv2.GaussianBlur(img, (3,3), 0)

img = cv2.fastNlMeansDenoising(grayimg, None, 10, 7, 21)
ret,thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#kernel = np.ones((5,5), np.uint8)
#threshdilate = cv2.dilate(thresh, kernel, iterations = 1)
#thresherode = cv2.erode(threshdilate, kernel, iterations = 1)
#laplace = cv2.Laplacian(thresh, cv2.CV_8UC1)
#imagel, contoursl, hierarchyl = cv2.findContours(laplace,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contrsl = cv2.drawContours(colorimg, contoursl, -1, (0, 255, 0), 1)

sobely = cv2.Sobel(thresh,cv2.CV_8UC1,0,1,ksize=5)

image, contours, hierarchy = cv2.findContours(sobely,
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contrs = cv2.drawContours(colorimg, contours, -1, (0, 255, 255), 1)

#sobelx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(thresh,cv2.CV_64F,0,1,ksize=5)

#print(contrs==contrsl)

#cv2.imshow("filtered", img)
#cv2.imshow("thresh", thresh)
#cv2.imshow("erode", thresherode)
#cv2.imshow("laplace", laplace)
#cv2.imshow("contours laplace", contrsl)

cv2.imshow("contours", contrs)
#cv2.imshow("X", sobelx)
cv2.imshow("Y", sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()

