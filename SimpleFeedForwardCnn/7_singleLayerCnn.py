# lecture 7 of week 1
'''
add bias to conv and then apply relu or other acitvation function
stack different convolution layers
'''

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

import numpy as np 
import cv2
from pprint import pprint

def strided_convolution3d(image, kernel, bias=0, stride=1, pad=0):
    # (for numpy arrays)
    _, rowsImage, colsImage = image.shape 
    # rowsImage, colsImage, channelsImage= image.shape
    channelsKern, rowsKern, colsKern = kernel.shape

    rowsConv = int((rowsImage + 2*pad - rowsKern)//stride + 1)
    colsConv = int((colsImage + 2*pad - colsKern)//stride + 1)

    convImage = np.zeros((rowsConv, colsConv))
    for y in range(rowsConv):
        if stride*y+rowsKern > rowsImage:
            break # if filter is already past image, there is no point increasing y anymore
        for x in range(colsConv):
            if stride*x+colsKern > colsImage:
                break # if filter is already past image, there is no point increasing x anymore
            convImage[y,x] = np.sum(image[0:channelsKern, stride*y:stride*y+rowsKern, stride*x:stride*x+rowsKern ])
    
    return convImage+bias

def relu_activation(convImages):
    return np.maximum(convImages, 0)

def softmax_activation(convImages):
    diff = np.max(convImages)
    convImages -= diff
    expValues = np.exp(convImages)
    normBase = np.sum(expValues, axis=1, keepdims=True)
    normValues = expValues/normBase

    return normValues

image = np.array([[[9,7,6,8,5,8],
                   [7,5,6,1,0,6],
                   [1,1,6,7,3,9],
                   [3,0,6,9,8,3],
                   [9,6,7,1,2,8],
                   [8,9,1,4,7,5]],
                  
                  [[2,4,3,0,4,8],
                   [0,6,2,3,2,5],
                   [4,1,0,8,8,7],
                   [6,7,7,7,9,2],
                   [3,6,9,1,9,2],
                   [2,3,8,5,1,8]],
                  
                  [[7,3,5,1,1,7],
                   [6,2,1,4,9,6],
                   [0,3,1,1,3,5],
                   [7,3,5,2,1,0],
                   [1,1,7,5,3,3],
                   [7,1,4,9,2,5]]])

np.random.seed(0)

kernels = [ np.random.randint(-3,3,(3,3,3)) for i in range(3) ]
convImages = []

biases = [ np.random.uniform(-2,2) for i in range(3) ]

for kernel,bias in zip(kernels, biases):
    convImage = strided_convolution3d(image, kernel, bias)
    convImages.append(convImage)

'''
To check if relu activation is working
convImages = np.negative(convImages)
pprint(convImages) -> also after applying relu
'''
# apply relu activation
convImagesRelu = relu_activation(convImages)

# OR

# apply softmax activation
convImagesSoftmax = softmax_activation(convImages)