'''
Resnet implementation of only 2 residual units
One residual unit = 2 identity blocks + 1 skip connection
Followed by fully 7 connected layers
One final softmax layer
'''

# python libraries
import numpy as np
from pprint import pprint

# self scripts
from activations import Relu, Softmax, Sigmoid
from pooling import MaxPool2d
from dense import Dense, Flatten
from convolution import Convolution
from normalization import BatchNormalization
from layerArithmetic import Add

images = np.random.randint(1,10, (3,24, 24))

class Identity:
    def __init__(self, data, numFilters, filterDim, filterRange, biasRange, filterStride = 1, poolSize = 2, poolStride = 2):
        self.convLayer = Convolution(numFilters, filterDim, filterRange, biasRange,filterStride)
        self.normLayer = BatchNormalization(gamma=1, beta=0, eps=1e-5)
        self.reluLayer = Relu()
        self.input = data

    def performForwardBlock(self):
        self.convLayer.forwardConvLayer(self.input) # perform convolutions
        data = self.convLayer.convMaps
        
        self.normLayer.forwardBatchNormalization(data) # normalize data
        data = self.normLayer.output

        self.reluLayer.applyRelu(data) # pass through relu activation
        data = self.reluLayer.reluActivatedOutput

        self.output = data

# identity block A
identityBlockA = Identity(images, 4, (3,5,5), (-2,2), (-2,2))
identityBlockA.performForwardBlock()

# identity block B
depthFiltersB = identityBlockA.output.shape[0]
identityBlockB = Identity(identityBlockA.output, 16, (depthFiltersB,5,5), (-3,3), (-5,5))
identityBlockB.performForwardBlock()

# skip connection 1
depthFiltersC = identityBlockB.output.shape[0]
skipConvLayer1 = Convolution(16,(depthFiltersC, 9,9), (-2,2), (2,2), stride=1, padding=0)
skipConvLayer1.forwardConvLayer(images)

# apply batch normalization to skip connection 1 convMaps
skipNormLayer1 = BatchNormalization(1,0)
skipNormLayer1.forwardBatchNormalization(skipConvLayer1.convMaps)

# add skip connection data to identity block data and apply Relu
skipAddLayer1 = Add([identityBlockB.output, skipNormLayer1.output])
skipAddLayer1.addData()
skipReluLayer1 = Relu()
skipReluLayer1.applyRelu(skipAddLayer1.output)
newData = skipReluLayer1.reluActivatedOutput

# identity block C
identityBlockC = Identity(newData, 32, (depthFiltersC,5,5), (-2,2), (-2,2))
identityBlockC.performForwardBlock()

# identity block D
depthFiltersD = identityBlockC.output.shape[0]
identityBlockD = Identity(identityBlockC.output, 64, (depthFiltersD,5,5), (-3,3), (-5,5))
identityBlockD.performForwardBlock()

# skip connection 2
depthFiltersE = identityBlockB.output.shape[0]
skipConvLayer2 = Convolution(64,(depthFiltersC, 9,9), (-2,2), (2,2), stride=1, padding=0)
skipConvLayer2.forwardConvLayer(newData)

# apply batch normalization to skip connection 1 convMaps
skipNormLayer2 = BatchNormalization(1,0)
skipNormLayer2.forwardBatchNormalization(skipConvLayer2.convMaps)

# add skip connection data to identity block data and apply Relu
skipAddLayer2 = Add([identityBlockD.output, skipNormLayer2.output])
skipAddLayer2.addData()
skipReluLayer2 = Relu()
skipReluLayer2.applyRelu(skipAddLayer2.output)
newData = skipReluLayer2.reluActivatedOutput

# change to fully connected network

# dense layer 1 - flatten filters and apply relu
denseLayerOneFlat = Flatten(newData)
denseLayerOne = Dense(denseLayerOneFlat.output.size, 2048)
denseLayerOne.forwardDense(denseLayerOneFlat.output)

reluDenseLayerOne = Relu()
reluDenseLayerOne.applyRelu(denseLayerOne.output)

# dense layer 2 - apply relu
denseLayerTwo = Dense(reluDenseLayerOne.reluActivatedOutput.size, 1024)
denseLayerTwo.forwardDense(reluDenseLayerOne.reluActivatedOutput)

reluDenseLayerTwo = Relu()
reluDenseLayerTwo.applyRelu(denseLayerTwo.output)

# dense layer 3 - apply relu
denseLayerThree = Dense(reluDenseLayerTwo.reluActivatedOutput.size, 512)
denseLayerThree.forwardDense(reluDenseLayerTwo.reluActivatedOutput)

reluDenseLayerThree = Relu()
reluDenseLayerThree.applyRelu(denseLayerThree.output)

# dense layer 4 - apply relu
denseLayerFour = Dense(reluDenseLayerThree.reluActivatedOutput.size, 256)
denseLayerFour.forwardDense(reluDenseLayerThree.reluActivatedOutput)

reluDenseLayerFour = Relu()
reluDenseLayerFour.applyRelu(denseLayerFour.output)

# dense layer 5 - apply relu
denseLayerFive = Dense(reluDenseLayerFour.reluActivatedOutput.size, 128)
denseLayerFive.forwardDense(reluDenseLayerFour.reluActivatedOutput)

reluDenseLayerFive = Relu()
reluDenseLayerFive.applyRelu(denseLayerFive.output)

# dense layer 6 - apply relu
denseLayerSix = Dense(reluDenseLayerFive.reluActivatedOutput.size, 64)
denseLayerSix.forwardDense(reluDenseLayerFive.reluActivatedOutput)

reluDenseLayerSix = Relu()
reluDenseLayerSix.applyRelu(denseLayerSix.output)

# dense layer 7 - apply relu
denseLayerSeven = Dense(reluDenseLayerSix.reluActivatedOutput.size, 10)
denseLayerSeven.forwardDense(reluDenseLayerSix.reluActivatedOutput)

sigmoidFinal = Sigmoid()
sigmoidFinal.applySigmoid(denseLayerSeven.output)

# Final Class Predictions 
classPredictions = sigmoidFinal.sigmoidActivatedOutput
classPredicted = np.argmax(classPredictions)
print(classPredictions)
print("The class predicted for the given image is : ", classPredicted)