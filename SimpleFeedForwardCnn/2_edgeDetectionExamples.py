# edge detection example in week1 lecture 2 of course

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

import numpy as np 
import cv2

image1 = np.array(([3,0,1,2,7,4],
                  [1,5,8,9,3,1],
                  [2,7,2,5,1,3],
                  [0,1,3,1,7,8],
                  [4,2,1,6,2,8],
                  [2,4,5,2,3,9]))
# print(image.shape)

image2 = np.array(([10,10,10,0,0,0],
                   [10,10,10,0,0,0], 
                   [10,10,10,0,0,0],
                   [10,10,10,0,0,0],
                   [10,10,10,0,0,0],
                   [10,10,10,0,0,0]))

kernel = np.array(([1,0,-1],
                   [1,0,-1],
                   [1,0,-1]))  
print(kernel.shape)

image = cv2.cvtColor(cv2.imread('image.jpeg'), cv2.COLOR_BGR2GRAY)
stride = 1
pad = 0
xImage, yImage = image.shape
xKern, yKern = kernel.shape

xConv = int((xImage + 2*pad - xKern)//stride + 1)
yConv = int((yImage + 2*pad - yKern)//stride + 1)
convImage = np.zeros((yConv, xConv))

for y in range(yConv):
    for x in range(xConv):
        convImage[y][x] = np.sum(image[y:y+yKern, x:x+xKern]*kernel)


cv2.imshow('convImage')
cv2.waitKey(0)
cv2.destroyAllWindows()