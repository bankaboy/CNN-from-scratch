# week 1 lecture 3 of course
# positive edges - light to dark transitions
# negative edges - dark to light transitions

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

import numpy as np 
import matplotlib.pyplot as plt 

image1 = np.array(([10,10,10,0,0,0],
                   [10,10,10,0,0,0], 
                   [10,10,10,0,0,0],
                   [10,10,10,0,0,0],
                   [10,10,10,0,0,0],
                   [10,10,10,0,0,0]))
print(image1)
plt.plot(image1)
plt.show()

image2 = image1.T
plt.plot(image2)
plt.show()


#normal
vertical_kernel = np.array(([1,0,-1],
                            [1,0,-1],
                            [1,0,-1]))  
'''
#sobel
vertical_kernel = np.array(([1,0,-1],
                            [2,0,-2],
                            [1,0,-1]))  
#scharr 
vertical_kernel = np.array(([3,0,-3],
                            [10,0,-10],
                            [3,0,-3]))  
'''
plt.plot(vertical_kernel)
plt.show()

horizontal_kernel = vertical_kernel.T
plt.plot(horizontal_kernel)
plt.show()

stride = 1
pad = 0
xImage, yImage = image1.shape
xKern, yKern = vertical_kernel.shape

xConv = int((xImage + 2*pad - xKern)//stride + 1)
yConv = int((yImage + 2*pad - yKern)//stride + 1)
vertical_convImage1 = np.zeros((yConv, xConv))
vertical_convImage2 = np.zeros((yConv, xConv))
horizontal_convImage1 = np.zeros((yConv, xConv))
horizontal_convImage2 = np.zeros((yConv, xConv))

for y in range(yConv):
    for x in range(xConv):
        vertical_convImage1[y][x] = np.sum(image1[y:y+yKern, x:x+xKern]*vertical_kernel)
        vertical_convImage2[y][x] = np.sum(image2[y:y+yKern, x:x+xKern]*vertical_kernel)
        horizontal_convImage1[y][x] = np.sum(image1[y:y+yKern, x:x+xKern]*horizontal_kernel)
        horizontal_convImage2[y][x] = np.sum(image2[y:y+yKern, x:x+xKern]*horizontal_kernel)


plt.plot(vertical_convImage1)
plt.show()

plt.plot(vertical_convImage2)
plt.show()

plt.plot(horizontal_convImage1)
plt.show()

plt.plot(horizontal_convImage2)
plt.show()