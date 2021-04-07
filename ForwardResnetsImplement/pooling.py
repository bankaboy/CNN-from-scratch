import numpy as np
import math 

class MaxPool2d:
    def __init__(self, filterSize = 2, stride = 2):
        self.poolMaps = []
        self.filterSize = filterSize
        self.stride = stride

    def maxPool(self, convMap):
        rowsOld, colsOld = convMap.shape
        rowsPool = math.floor((rowsOld-self.filterSize)/self.stride + 1)
        colsPool = math.floor((colsOld-self.filterSize)/self.stride + 1)

        poolMap = np.zeros((rowsPool, colsPool))
        for y in range(rowsPool):
            if self.stride*y+self.filterSize > rowsOld:
                break # if filter is already past image, there is no point increasing y anymore
            for x in range(colsPool):
                if self.stride*x+self.filterSize > colsOld:
                    break # if filter is already past image, there is no point increasing x anymore
                poolMap[y,x] = np.max(convMap[self.stride*y:self.stride*y+self.filterSize, self.stride*x:self.stride*x+self.filterSize])

        return poolMap

    def forwardMaxPoolLayer(self, convMaps):
        for convMap in convMaps:
            self.poolMaps.append(self.maxPool(convMap))
        self.poolMaps = np.array(self.poolMaps)

class AvgPool2d:
    def __init__(self, filterSize = 2, stride = 2):
        self.poolMaps = []
        self.filterSize = filterSize
        self.stride = stride

    def avgPool(self, convMap):
        rowsOld, colsOld = convMap.shape
        rowsPool = math.floor((rowsOld-self.filterSize)/self.stride + 1)
        colsPool = math.floor((colsOld-self.filterSize)/self.stride + 1)

        poolMap = np.zeros((rowsPool, colsPool))
        for y in range(rowsPool):
            if self.stride*y+self.filterSize > rowsOld:
                break # if filter is already past image, there is no point increasing y anymore
            for x in range(colsPool):
                if self.stride*x+self.filterSize > colsOld:
                    break # if filter is already past image, there is no point increasing x anymore
                poolMap[y,x] = np.average(convMap[self.stride*y:self.stride*y+self.filterSize, self.stride*x:self.stride*x+self.filterSize])

        return poolMap

    def forwardAvgPoolLayer(self, convMaps):
        for convMap in convMaps:
            self.poolMaps.append(self.avgPool(convMap))
        self.poolMaps = np.array(self.poolMaps)