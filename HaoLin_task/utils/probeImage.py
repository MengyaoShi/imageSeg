import numpy as np
import cv2
from matplotlib import pyplot as plt

def probeImage(input=None,xy=[10,13]):
    '''
    the input can be none, an image array or a path for an image
    if the xy is relative, scale to 224
    '''
    img = np.random.rand(28,28,3)
    if isinstance(input, np.ndarray):
        img = input
    else:
        img = cv2.imread(input)
    if xy[0]<1:
        xy[0] =  xy[0]*224 #image width
        xy[1] = xy[1]*224
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.scatter(xy[0],xy[1], marker = 'x', color = "red", s = 15)
    plt.show()

if __name__== "__main__":
    probeImage()

