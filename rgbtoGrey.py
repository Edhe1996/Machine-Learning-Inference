import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.misc import imread


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


img = imread('color1.jpg')
gray = rgb2gray(img)
misc.imsave('gray1.jpg', gray)
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.show()
