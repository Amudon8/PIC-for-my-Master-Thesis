import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('10000particles2.png')
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(grayscale))
plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.imshow(np.log(abs(dark_image_grey_fourier)))
plt.title('Dispersion Diagram', fontsize=20)
plt.ylabel('omega-space', fontsize=15)
plt.xlabel('k-space', fontsize=15)
plt.colorbar()
plt.show()