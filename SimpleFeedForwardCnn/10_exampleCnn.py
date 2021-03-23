# Lecture 10 of week 1

'''
Change code accordingly
NUMPY NOTATION : C x H x W (if using random np array as test image)
CV2 NOTATION : H x W x C (if using actual image and using cv2 to load)
'''

'''
input is 32x32x3, filter is 5x5 and stride is 1 and 6 filters, and apply relu. 28x28x6
then apply pooling layer , max pooling , f=2, s=2, now 14x14x6 (conv1)
apply another filter to conv1, 16 filters, f=5, s=1
then apply max pooling, f=2, s=2, now 5x5x16 (conv2)'
flatten pool2 400 units and connect to layer with 120 units (fully connected layer)
add another layer 84 units, connect to softmax unit , 10 outputs
'''


import numpy as np
import math
from pprint import pprint

# class layerConvolution
class layerConvolution:
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



# class pooling:
class pooling:
    def __init__(self, filterSize = 2, stride = 2, poolType = "MAX"):
        self.poolMaps = []
        self.filterSize = filterSize
        self.stride = stride
        self.poolType = poolType

    def maxPooling(self, convMap):
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

    def avgPooling(self, convMap):
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

    def forwardPoolLayer(self, convMaps):

        if self.poolType == "MAX":
            for convMap in convMaps:
                self.poolMaps.append(self.maxPooling(convMap))

        elif self.poolType == "AVERAGE":
            for convMap in convMaps:
                self.poolMaps.append(self.avgPooling(convMap))

        self.poolMaps = np.array(self.poolMaps)

# class layerDense:
class layerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forwardDense(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def flattenConvMap(self, convMaps):
        convMaps = convMaps.flatten()
        return convMaps


# class reluActivation:
class reluActivation:
    def applyRelu(self, arr):
        self.reluActivatedOutput = np.maximum(arr, 0)


# class softmaxActivation:
class softmaxActivation:
    def applySoftmax(self, arr):
        arr -= np.max(arr)
        exp_values = np.exp(arr)
        self.softmaxActivatedOutput = exp_values/np.sum(exp_values, axis=1, keepdims=True)


# class sigmoidActivation:
class sigmoidActivation:
    def applySigmoid(self,arr):
        self.sigmoidActivatedOutput = 1/(1 + np.exp(-arr))



np.random.seed(0)
image = np.random.randint(1,10,(3,32,32))


# Convolution Layer 1
convLayerOne = layerConvolution(numFilters=6, filterDim=(1,3,3), filterRange=(-2,2), biasRange=(-2,2), stride=1, padding=0)
convLayerOne.forwardConvLayer(image=image)

# Apply ReLU on convolution Layer One (can replace with softmax)
reluConvLayerOne = reluActivation()
reluConvLayerOne.applyRelu(convLayerOne.convMaps)

# Apply Max Pooling on activated convolution maps of convolutional layer one
poolingLayerOne = pooling(2,2,"MAX")
poolingLayerOne.forwardPoolLayer(reluConvLayerOne.reluActivatedOutput)

# Convolution Layer Two
convLayerTwo = layerConvolution(numFilters=16, filterDim=(poolingLayerOne.poolMaps.shape[0],3,3), filterRange=(-2,2), biasRange=(-2,2), stride=1, padding=0)
convLayerTwo.forwardConvLayer(image=poolingLayerOne.poolMaps)

# Apply ReLU on convolution Layer Two (can replace with softmax)
reluConvLayerTwo = reluActivation()
reluConvLayerTwo.applyRelu(convLayerTwo.convMaps)

# Apply Max Pooling on activated convolution maps of convolutional layer Two
poolingLayerTwo = pooling(2,2,"MAX")
poolingLayerTwo.forwardPoolLayer(reluConvLayerTwo.reluActivatedOutput)

# Dense Layer One - Flatten filters
denseLayerOneInputSize = poolingLayerTwo.poolMaps.size
denseLayerOne = layerDense(denseLayerOneInputSize, 120)
denseLayerOneInputs = denseLayerOne.flattenConvMap(poolingLayerTwo.poolMaps)
denseLayerOne.forwardDense(denseLayerOneInputs)

# Apply Relu to dense layer one
reluDenseLayerOne = reluActivation()
reluDenseLayerOne.applyRelu(denseLayerOne.output)

# Dense Layer Two
denseLayerTwoInputSize = reluDenseLayerOne.reluActivatedOutput.size
denseLayerTwo = layerDense(denseLayerTwoInputSize, 84)
denseLayerTwoInputs = reluDenseLayerOne.reluActivatedOutput
denseLayerTwo.forwardDense(denseLayerTwoInputs)

# Apply Relu to dense layer two
reluDenseLayerTwo = reluActivation()
reluDenseLayerTwo.applyRelu(denseLayerTwo.output)

# Dense Layer Three
denseLayerThreeInputSize = reluDenseLayerTwo.reluActivatedOutput.size
denseLayerThree = layerDense(denseLayerThreeInputSize, 10)
denseLayerThreeInputs = reluDenseLayerTwo.reluActivatedOutput
denseLayerThree.forwardDense(denseLayerThreeInputs)

'''
Decided to apply softmax as sigmoid led to overflow
# Apply Sigmoid to outputs of dense layer three
sigmoidFinal = sigmoidActivation()
sigmoidFinal.applySigmoid(denseLayerThree.output)
pprint(sigmoidFinal.sigmoidActivatedOutput)
'''

# Apply softmax to outputs of dense layer three
softmaxFinal = softmaxActivation()
softmaxFinal.applySoftmax(denseLayerThree.output)
# pprint(softmaxFinal.softmaxActivatedOutput)

# Final Class Predictions 
classPredictions = softmaxFinal.softmaxActivatedOutput
classPredicted = np.argmax(classPredictions)
print("The class predicted for the given image is : ", classPredicted)