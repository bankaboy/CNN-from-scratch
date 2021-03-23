# Lecture 6 of week 1 cnn coursera
'''
Need 3d filter for R,G,B channels
H x W x C
number of channels in filter has to be equal to number of channels in image
produce 2d convmap after
'''

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

import numpy as np
import cv2

def strided_convolution3d(image, kernel, stride=1, pad=0):
    # channelsImage, rowsImage, colsImage = image.shape (for numpy arrays)
    rowsImage, colsImage, channelsImage= image.shape
    channelsKern, rowsKern, colsKern = kernel.shape

    rowsConv = int((rowsImage + 2*pad - rowsKern)//stride + 1)
    colsConv = int((colsImage + 2*pad - colsKern)//stride + 1)
    print(rowsImage, colsImage, rowsConv, colsConv)
    convImage = np.zeros((rowsConv, colsConv))
    for y in range(rowsConv):
        if stride*y+rowsKern > rowsImage:
            break # if filter is already past image, there is no point increasing y anymore
        for x in range(colsConv):
            if stride*x+colsKern > colsImage:
                break # if filter is already past image, there is no point increasing x anymore
            convImage[y,x] = np.sum(image[stride*y:stride*y+rowsKern, stride*x:stride*x+rowsKern, 0:channelsKern])

    return convImage

# convImage[y][x] = np.sum(image[0:channelsImage, stride*y:stride*y+colsKern, stride*x:stride*x+rowsKern]*kernel) (for random numpy array)

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

kernel = np.array([[[1,0,-1],
                    [1,0,-1],
                    [1,0,-1]],
                  
                   [[1,0,-1],
                    [1,0,-1],
                    [1,0,-1]],
                  
                   [[1,0,-1],
                    [1,0,-1],
                    [1,0,-1]]]) 

image1 = cv2.imread('image.jpeg')
convImage = strided_convolution3d(image1, kernel)%255
# print(convImage)
cv2.imshow('After Conv', convImage)
cv2.waitKey(0)
cv2.destroyAllWindows()