import numpy as np

class Convolution:
    def __init__(self, numFilters, filterDim, filterRange, biasRange, stride = 1, padding = 0):
        self.numFilters = numFilters
        self.filters = [ np.random.randint(filterRange[0], filterRange[1], (filterDim)) for i in range(numFilters)]
        self.biases = [ np.random.uniform(filterRange[0], filterRange[1]) for i in range(numFilters)]
        self.stride = stride
        self.pad = padding
        self.convMaps = []

    def strided_convolution3d(self, image, filter, bias):
        # (for numpy arrays)
        _, rowsImage, colsImage = image.shape 
        # rowsImage, colsImage, channelsImage= image.shape
        channelsFilter, rowsFilter, colsFilter = filter.shape

        rowsConv = int((rowsImage + 2*self.pad - rowsFilter)//self.stride + 1)
        colsConv = int((colsImage + 2*self.pad - colsFilter)//self.stride + 1)

        convImage = np.zeros((rowsConv, colsConv))
        for y in range(rowsConv):
            if self.stride*y+rowsFilter > rowsImage:
                break # if filter is already past image, there is no point increasing y anymore
            for x in range(colsConv):
                if self.stride*x+colsFilter > colsImage:
                    break # if filter is already past image, there is no point increasing x anymore
                convImage[y,x] = np.sum(image[0:channelsFilter, self.stride*y:self.stride*y+rowsFilter, self.stride*x:self.stride*x+rowsFilter ])

        return convImage+bias
        

    def forwardConvLayer(self, image):
        for filter, bias in zip(self.filters, self.biases):
            self.convMaps.append(self.strided_convolution3d(image, filter, bias))
        self.convMaps = np.array(self.convMaps)