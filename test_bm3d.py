import numpy as np
import skimage.data
from skimage.measure import compare_psnr

import pybm3d
import cv2


noise_std_dev = 40
img = skimage.data.astronaut()
noise = np.random.normal(scale=noise_std_dev,
                         size=img.shape).astype(img.dtype)

noisy_img = img + noise

out = pybm3d.bm3d.bm3d(noisy_img, noise_std_dev)

#noise_psnr = compare_psnr(img, noisy_img)
#out_psnr = compare_psnr(img, out)

#print("PSNR of noisy image: ", noise_psnr)
#print("PSNR of reconstructed image: ", out_psnr)

cv2.imshow("img_no_noise", out)
cv2.imshow("img_noise", noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
