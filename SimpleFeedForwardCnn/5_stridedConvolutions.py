# Week 1 lecture 5

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

import numpy as np
import cv2

def strided_convolution(image, kernel, stride=1, pad=0):
    xImage, yImage = image.shape
    xKern, yKern = kernel.shape

    xConv = int((xImage + 2*pad - xKern)//stride + 1)
    yConv = int((yImage + 2*pad - yKern)//stride + 1)
    convImage = np.zeros((yConv, xConv))
    for y in range(0, yConv):
        if stride*y+yKern > yImage: # skip iteration if kernel hangs outside image
            break
        for x in range(0, xConv):
            if stride*x+xKern > xImage: # skip iteration if kernel hangs outside image
                break
            convImage[y][x] = np.sum(image[stride*y:stride*y+yKern, stride*x:stride*x+xKern]*kernel)

    return convImage

testArray = np.array(([2,3,7,4,6,2,9],
                      [6,6,9,8,7,4,3],
                      [3,4,8,3,8,9,7],
                      [7,8,3,6,6,3,4],
                      [4,2,1,8,3,4,6],
                      [3,2,4,1,9,8,3],
                      [0,1,3,9,2,1,4]))

kernel = np.array(([3,4,4],
                   [1,0,2],
                   [-1,0,3]))

convImage = strided_convolution(image=testArray, kernel=kernel, stride=2)
print(convImage)