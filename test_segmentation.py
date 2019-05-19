import cv2
import numpy as np
from scipy import ndimage

colorimg = cv2.imread("1.jpeg", 1)
grayimg = cv2.cvtColor(colorimg, cv2.COLOR_BGR2GRAY)

img = cv2.fastNlMeansDenoising(grayimg, None, 10, 7, 21)
ret,thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

sobely = cv2.Sobel(thresh,cv2.CV_8UC1,0,1,ksize=5)

image, contours, hierarchy = cv2.findContours(sobely,
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contrs = cv2.drawContours(colorimg, contours, -1, (0, 255, 255), 1)

cv2.imshow("contours", contrs)
cv2.imshow("Y", sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()

