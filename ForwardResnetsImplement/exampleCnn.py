import numpy as np

from activations import Relu, Softmax
from pooling import MaxPool2d
from dense import Dense, Flatten
from convolution import Convolution

np.random.seed(0)
image = np.random.randint(1,10,(3,32,32))


# Convolution Layer 1
convLayerOne = Convolution(numFilters=6, filterDim=(1,3,3), filterRange=(-2,2), biasRange=(-2,2), stride=1, padding=0)
convLayerOne.forwardConvLayer(image=image)

# Apply ReLU on convolution Layer One (can replace with softmax)
reluConvLayerOne = Relu()
reluConvLayerOne.applyRelu(convLayerOne.convMaps)

# Apply Max MaxPool2d on activated convolution maps of convolutional layer one
MaxPool2dLayerOne = MaxPool2d(2,2)
MaxPool2dLayerOne.forwardMaxPoolLayer(reluConvLayerOne.reluActivatedOutput)

# Convolution Layer Two
convLayerTwo = Convolution(numFilters=16, filterDim=(MaxPool2dLayerOne.poolMaps.shape[0],3,3), filterRange=(-2,2), biasRange=(-2,2), stride=1, padding=0)
convLayerTwo.forwardConvLayer(image=MaxPool2dLayerOne.poolMaps)

# Apply ReLU on convolution Layer Two (can replace with softmax)
reluConvLayerTwo = Relu()
reluConvLayerTwo.applyRelu(convLayerTwo.convMaps)

# Apply Max MaxPool2d on activated convolution maps of convolutional layer Two
MaxPool2dLayerTwo = MaxPool2d(2,2)
MaxPool2dLayerTwo.forwardMaxPoolLayer(reluConvLayerTwo.reluActivatedOutput)

# Dense Layer One - Flatten filters
denseLayerOneInputSize = MaxPool2dLayerTwo.poolMaps.size
denseLayerOne = Dense(denseLayerOneInputSize, 120)
flattenLayer = Flatten(MaxPool2dLayerTwo.poolMaps)
denseLayerOneInputs = flattenLayer.output
denseLayerOne.forwardDense(denseLayerOneInputs)

# Apply Relu to dense layer one
reluDenseLayerOne = Relu()
reluDenseLayerOne.applyRelu(denseLayerOne.output)

# Dense Layer Two
denseLayerTwoInputSize = reluDenseLayerOne.reluActivatedOutput.size
denseLayerTwo = Dense(denseLayerTwoInputSize, 84)
denseLayerTwoInputs = reluDenseLayerOne.reluActivatedOutput
denseLayerTwo.forwardDense(denseLayerTwoInputs)

# Apply Relu to dense layer two
reluDenseLayerTwo = Relu()
reluDenseLayerTwo.applyRelu(denseLayerTwo.output)

# Dense Layer Three
denseLayerThreeInputSize = reluDenseLayerTwo.reluActivatedOutput.size
denseLayerThree = Dense(denseLayerThreeInputSize, 10)
denseLayerThreeInputs = reluDenseLayerTwo.reluActivatedOutput
denseLayerThree.forwardDense(denseLayerThreeInputs)

# Apply softmax to outputs of dense layer three
softmaxFinal = Softmax()
softmaxFinal.applySoftmax(denseLayerThree.output)
# pprint(softmaxFinal.softmaxActivatedOutput)

# Final Class Predictions 
classPredictions = softmaxFinal.softmaxActivatedOutput
classPredicted = np.argmax(classPredictions)
print("The class predicted for the given image is : ", classPredicted)