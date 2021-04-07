import numpy as np

class Add():
  def __init__(self, inputs):
    self.inputs = inputs
  def addData(self):
    data = self.inputs[0]
    for i in range(1, len(self.inputs)):
      data += self.inputs[i]
    self.output = data

class Subtract():
  def _merge_function(self, inputs):
    if len(inputs) != 2:
      raise ValueError('A `Subtract` layer should be called '
                       'on exactly 2 inputs')
    return inputs[0] - inputs[1]

class Multiply():
  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = output * inputs[i]
    return output

class Average():
  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output += inputs[i]
    return output / len(inputs)

class Maximum():
  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = np.maximum(output, inputs[i])
    return output


class Minimum():
  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = np.minimum(output, inputs[i])
    return output

# class Concatenate():
