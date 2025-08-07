import numpy as np

# Sets any negative value to 0
class ReLU:
    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # set all negative numbers to zero
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        dinputs = dvalues * (np.array(self.inputs) > 0)
        del self.inputs
        return dinputs


# allows negative numbers, divided by 10 or * 0.1
class LeakyReLU:
    # forward pass, setting negative numbers to /10
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0.1 * inputs, inputs)

    # backward pass for LeakyReLU
    def backward(self, dvalues):
        dinputs = dvalues * (np.array(self.inputs) > 0) + 0.1 * dvalues * (np.array(self.inputs) < 0)
        del self.inputs
        return dinputs


# Exponential Linear Unit
class ELU:
    # set alpha value
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    # forward method
    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # return ELU function
        return np.where(inputs >= 0, inputs, self.alpha * np.exp(inputs) - 1)

    # backward pass
    def backward(self, dvalues):
        #calc the derivative
        deriv = np.where(self.inputs >= 0, 1, self.alpha * np.exp(self.inputs))
        # free self.inputs to save memory
        del self.inputs
        #return grads
        return deriv * dvalues


# scales the values between 1 & 0
class Sigmoid:
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, dvalues):
        dinputs = self.outputs * (1 - self.outputs) * dvalues
        del self.outputs
        return dinputs


# Scales the values between -1 & 1
class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self, dvalues):
        dvalues = dvalues * (1 - np.tanh(self.inputs) ** 2)
        del self.inputs
        return dvalues


# activation function for output layer
class Softmax:
    def forward(self, inputs):
        # Normilize inputs to prevent overflowing by subtracting maxium in the batch
        norm_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_inputs = np.exp(norm_inputs)
        # divide eulur's number to the i by the sum of all i values in the batch
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.outputs

    def backward(self, dvalues):
        dinputs = np.empty_like(self.outputs)
        for i, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            softmax_jacobian = np.diag(single_output) - np.outer(single_output, single_output)
            dinputs[i] = np.dot(softmax_jacobian, single_dvalues)
        del self.outputs
        return dinputs
