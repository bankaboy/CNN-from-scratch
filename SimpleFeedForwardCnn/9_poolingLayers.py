# Lecture 9 of week 1

'''
max pooling : 
take max of each region
region is determined by filter size and stride
large number in convMap means it has detected certain feature
fixed computation, no parameters to learn
perform on every conv map after filter
'''

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

import numpy as np
import cv2
import math
from pprint import pprint

def maxPooling(convMap, filterSize = 2, stride = 2):
    rowsOld, colsOld = convMap.shape
    rowsPool = math.floor((rowsOld-filterSize)/stride + 1)
    colsPool = math.floor((colsOld-filterSize)/stride + 1)

    poolMap = np.zeros((rowsPool, colsPool))
    for y in range(rowsPool):
        if stride*y+filterSize > rowsOld:
            break # if filter is already past image, there is no point increasing y anymore
        for x in range(colsPool):
            if stride*x+filterSize > colsOld:
                break # if filter is already past image, there is no point increasing x anymore
            poolMap[y,x] = np.max(convMap[stride*y:stride*y+filterSize, stride*x:stride*x+filterSize])
    
    return poolMap

def avgPooling(convMap, filterSize = 2, stride = 2):
    rowsOld, colsOld = convMap.shape
    rowsPool = math.floor((rowsOld-filterSize)/stride + 1)
    colsPool = math.floor((colsOld-filterSize)/stride + 1)

    poolMap = np.zeros((rowsPool, colsPool))
    for y in range(rowsPool):
        if stride*y+filterSize > rowsOld:
            break # if filter is already past image, there is no point increasing y anymore
        for x in range(colsPool):
            if stride*x+filterSize > colsOld:
                break # if filter is already past image, there is no point increasing x anymore
            poolMap[y,x] = np.average(convMap[stride*y:stride*y+filterSize, stride*x:stride*x+filterSize])
    
    return poolMap    

convMap1 = np.array(([1,3,2,1],
                    [2,9,1,1],
                    [1,3,2,3],
                    [5,6,1,2]))
        
convMap2 = np.array(([1,3,2,1,3],
                     [2,9,1,1,5],
                     [1,3,2,3,2],
                     [8,3,5,1,0],
                     [5,6,1,2,9]))
pprint(convMap1)
maxPoolMap = maxPooling(convMap1,filterSize=2,stride=2)
avgPoolMap = avgPooling(convMap1,filterSize=2,stride=2)
pprint(avgPoolMap)