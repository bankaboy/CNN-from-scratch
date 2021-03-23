# video 4 of lecture week 1
'''
image shrinks on every convolution
pixels on the edges are used in convolution very less that the middle pixels.
information of edge pixels is lost
'''
# https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/#what-is-constant-padding
# Constant, Reflection, Replication

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

import numpy as np
import cv2

def padding(image, amount = 0, type = 'CONSTANT'):
    xImageOld, yImageOld = image.shape
    xImageNew = xImageOld + 2*amount
    yImageNew = yImageOld + 2*amount
    newImage = np.zeros((xImageNew, yImageNew)) 
    if type == 'CONSTANT':
        newImage[amount:amount+xImageOld, amount:amount+yImageOld] = image


    return newImage


image = cv2.cvtColor(cv2.imread('image.jpeg'), cv2.COLOR_BGR2GRAY)
paddedImage = padding(image = image, amount = 5, type = 'CONSTANT')
cv2.imshow('Original', image)
cv2.imshow('Padded', paddedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()