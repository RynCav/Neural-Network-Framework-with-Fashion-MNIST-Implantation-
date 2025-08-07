from numba import cuda
import math
from ._weight_init import *
from . import _gpu_helper as gpu


# a single dense layer of a neural network
class DenseLayer:
    def __init__(self, n_inputs, n_neurons, innitlized = he, l1=0, l2=0):
        self.weights = innitlized((n_inputs, n_neurons))

        # set all biases in array to standard 0 for each neuron
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # multiply the inputs by the weights and add the biases for each batch
        outputs = inputs @ self.weights + self.biases
        return outputs

    def backward(self, dvalues):
        # calc the  partial derivative of each value
        self.dweights = np.dot(np.array(self.inputs).T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        del self.inputs
        return self.dinputs


# a convultional layer
class ConvolutionLayer:
    def __init__(self, kernels, height, width, depth, stride=1, p=0):
        self.height = height
        self.width = width
        self.depth = depth
        self.stride = stride
        self.p = p

        # Weight initialization
        shape = (kernels, depth, height, width)
        self.weights = np.random.randn(*shape).astype(np.float32)
        self.biases = np.zeros(kernels)

    def forward(self, inputs):

        if inputs.ndim == 3:
            inputs = inputs[:, np.newaxis, :, :]

        if self.p:
            inputs = np.pad(inputs,((0, 0), (0, 0), (self.p, self.p), (self.p, self.p)), mode='constant')

        self.inputs = inputs.astype(np.float32)
        self.shape = inputs.shape

        new_height = (self.shape[2] - self.height) // self.stride + 1
        new_width = (self.shape[3] - self.width) // self.stride + 1

        convoluted = np.zeros((self.shape[0], self.weights.shape[0], new_height, new_width)).astype(np.float32)

        gpu_array = cuda.to_device(convoluted)
        gpu_weights = cuda.to_device(self.weights)
        gpu_inputs = cuda.to_device(inputs)



        threads = (16, 16)
        blocks_x = max(1, math.ceil(new_width / threads[0]))
        blocks_y = max(1, math.ceil(new_height / threads[1]))
        blocks_z = max(1, inputs.shape[0] * self.weights.shape[0])

        blocks = (blocks_x, blocks_y, blocks_z)

        gpu.convolve[blocks, threads](gpu_array, gpu_weights, gpu_inputs, self.stride)
        outputs = gpu_array.copy_to_host()

        del gpu_array
        del gpu_weights
        del gpu_inputs


        return outputs + self.biases.reshape(1, -1, 1, 1)

    def backward(self, dvalues):

        self.dbiases = np.sum(dvalues, (0, 2, 3))
        self.dweights = np.zeros_like(self.weights)
        self.dinputs = np.zeros_like(self.inputs)

        flippedWeights = self.weights[:, :, ::-1, ::-1]

        self.dweights, self.dinputs = gpu.dconvolve(self.dweights, flippedWeights, self.dinputs, dvalues, self.inputs, self.stride)


        if self.p:
            return self.dinputs[:, :, self.p:-self.p, self.p:-self.p]

        return self.dinputs


# max pooling layer
class PoolingLayer:
    def __init__(self, height, width, stride):
        self.dims_height, self.dims_width = height, width
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, channels, height, width = inputs.shape

        height_out = (height - self.dims_height) // self.stride + 1
        width_out = (width - self.dims_width) // self.stride + 1

        output = np.zeros((batch_size, channels, height_out, width_out))
        self.pooled_idx = np.zeros_like(output, dtype=int)

        for i in range(batch_size):
            for c in range(channels):
                for h in range(height_out):
                    for w in range(width_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        window = inputs[i, c, h_start:h_start + self.dims_height, w_start:w_start + self.dims_width]
                        output[i, c, h, w] = np.max(window)
                        self.pooled_idx[i, c, h, w] = np.argmax(window)
        return output

    def backward(self, dvalues):
        dinputs = np.zeros_like(self.inputs)

        batch_size, channels, height_out, width_out = dvalues.shape

        for i in range(batch_size):
            for c in range(channels):
                for h in range(height_out):
                    for w in range(width_out):
                        h_start = h * self.stride
                        w_start = w * self.stride

                        window = self.inputs[i, c, h_start:h_start + self.dims_height,
                                 w_start:w_start + self.dims_width]
                        max_idx = self.pooled_idx[i, c, h, w]
                        max_pos = np.unravel_index(max_idx, window.shape)

                        dinputs[i, c, h_start + max_pos[0], w_start + max_pos[1]] += dvalues[i, c, h, w]
        return dinputs


# fractionally strided convolutional layer
class TransposedConvolutionLayer:
    def __init__(self, kernels, height, width, depth, stride=1):
        self.height = height
        self.width = width
        self.depth = depth
        self.kernels = kernels
        self.stride = stride

        # Weight initialization
        shape = (kernels, depth, height, width)
        self.weights = np.random.randn(*shape)

        self.biases = np.zeros(kernels)

    def forward(self, inputs):

        if inputs.ndim == 3:
            inputs = inputs[:, np.newaxis, :, :]

        self.inputs = inputs

        newHeight = (inputs.shape[2] - 1) * self.stride + self.height
        newWidth = (inputs.shape[3] - 1) * self.stride +  self.width

        convoluted = np.zeros((inputs.shape[0], self.kernels, newHeight, newWidth))

        batch, depth, height, width = inputs.shape

        for i in range(batch):
            for n in range(depth):
                for h in range(height):
                    for w in range(width):

                        hStart = h * self.stride
                        wStart = w * self.stride

                        hEnd = hStart + self.height
                        wEnd = wStart + self.width

                        for k, kernel in enumerate(self.kernels):
                            convoluted[i, k, hStart:hEnd, wStart:wEnd] += inputs[i, n, h, w] * kernel[n]

        return convoluted + self.biases.reshape(1, -1, 1, 1)

    def backward(self, dvalues):

        self.dbiases = np.sum(dvalues, (0, 2, 3))
        self.dweights = np.zeros_like(self.weights)
        self.dinputs = np.zeros_like(self.inputs)

        batch, depth, height, width = self.inputs.shape

        flippedWeights = self.weights[:, :, ::-1, ::-1]

        for i in range(batch):
            for k, kernel in enumerate(flippedWeights):
                for h in range(height):
                    for w in range(width):

                        hStart = h * self.stride
                        wStart = w * self.stride
                        hEnd = hStart + self.height
                        wEnd = wStart + self.width

                        for n in range(depth):
                            self.dweights[k, n] += dvalues[i, k, h, w] * self.inputs[i, n, hStart:hEnd, wStart:wEnd]
                            self.dinputs[i, n, hStart:hEnd, wStart:wEnd] += dvalues[i, k, h, w] * kernel[n]

        return self.dinputs


# Dropout Layer FINISH
class DropoutLayer:
    def __init__(self, rate=0.3):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.mask = np.random.random(self.inputs.shape) > self.rate
        return inputs * self.mask

    def backward(self, dvalues):
        dvalues = dvalues * self.mask
        del self.mask, self.inputs
        return dvalues


#normilizes inputs for stability
class BatchNormalization:
    def __init__(self, momentum=0.1, epsilon=1e-8):
        self.m = momentum
        self.epsilon = epsilon
        self.mean = None
        self.var = None

    def forward(self, inputs):
        #check if it is the first pass
        if self.mean is None:
            #calc current mean & var
            self.mean = np.mean(inputs, axis=0)
            self.var = np.var(inputs, axis=0)
        else:
            # update the mean & var
            self.mean = self.m * self.mean + (1 - self.m) * np.mean(inputs, axis=0)
            self.var = self.m * self.var + (1 - self.m) * np.var(inputs, axis=0)
        return (inputs - self.mean) / np.sqrt(self.var + self.epsilon)

    def backward(self, dinputs):
        return dinputs * (1 / np.sqrt(self.var + self.epsilon))


# layer that flattens tensors
class FlattenLayer:
    def forward(self, inputs):
        # stores the original shape for backward pass
        self.original_shape = np.array(inputs).shape
        # returns the reshaped matrix
        flattened_inputs = np.array(inputs).flatten()
        return np.split(flattened_inputs, self.original_shape[0])

    def backward(self, dinputs):
        return dinputs.reshape(self.original_shape)
