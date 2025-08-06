import numpy as np
import pickle as pkl
import math

from numpy.conftest import dtype

import helper
from numba import cuda
"""Helper functions for initializing weights"""
# for ReLU
def he(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])


# for other
def xavier(shape):
    return np.random.randn(*shape) * np.sqrt(1 / shape[0])


# for random
def normal(shape):
    return np.random.uniform(0, 0.1, shape)


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

        helper.convolve[blocks, threads](gpu_array, gpu_weights, gpu_inputs, self.stride)
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

        self.dweights, self.dinputs = helper.dconvolve(self.dweights, flippedWeights, self.dinputs, dvalues, self.inputs, self.stride)


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


# Categorical Cross-entropy with One Hot encoded data
class CCE:
     def forward(self, model, predicated, expected, epsilon=1e-8):
         # clip values to prevent log of 0 errors in each batch
         clipped = np.clip(predicated, epsilon, 1 - epsilon)
         loss = -np.sum(expected * np.log(clipped)) / expected.shape[0]
         return loss

     def backward(self, predicated, expected, epsilon=1e-8):
         clipped = np.clip(predicated, epsilon, 1 - epsilon)
         dinputs = (clipped - expected) / expected.shape[0]
         return dinputs


# Mean Squared Error
class MSE:
    def forward(self, model, predicted, expected):
        loss = np.sum(np.square(predicted - expected)) / len(expected)
        return loss

    def backward(self, predicted, expected):
        return (2 / len(expected)) * (predicted - expected)


# optimizer
class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.t = 0
        self.b1, self.b2 = b1, b2
        self.cached = {}

    def update(self, Layer, lr, epsilon=1e-8):
        # Initialize cache if not already done
        if id(Layer) not in self.cached:
            self.cached[id(Layer)] = {
                "mw": np.zeros_like(Layer.weights),
                "vw": np.zeros_like(Layer.weights),
                "mb": np.zeros_like(Layer.biases),
                "vb": np.zeros_like(Layer.biases)
           }

        # Increment step count
        self.t += 1

        # Retrieve cached momentums and velocities
        mw = self.cached[id(Layer)]["mw"]
        vw = self.cached[id(Layer)]["vw"]
        mb = self.cached[id(Layer)]["mb"]
        vb = self.cached[id(Layer)]["vb"]


        # Update momentums and velocities for weights
        mw = self.b1 * mw + (1 - self.b1) * Layer.dweights
        vw = self.b2 * vw + (1 - self.b2) * (Layer.dweights ** 2)

        # Update momentums and velocities for biases
        mb = self.b1 * mb + (1 - self.b1) * Layer.dbiases
        vb = self.b2 * vb + (1 - self.b2) * (Layer.dbiases ** 2)


        # Corrected momentums and velocities for weights
        mw_hat = mw / (1 - self.b1 ** self.t)
        vw_hat = vw / (1 - self.b2 ** self.t)
        mb_hat = mb / (1 - self.b1 ** self.t)
        vb_hat = vb / (1 - self.b2 ** self.t)

        # Update weights and biases
        Layer.weights = Layer.weights - lr * mw_hat / (np.sqrt(vw_hat) + epsilon)
        Layer.biases = Layer.biases - lr * mb_hat / (np.sqrt(vb_hat) + epsilon)

        # Save the updated momentums and velocities in the cache
        self.cached[id(Layer)] = {"mw": mw, "vw": vw, "mb": mb, "vb": vb}


# stops the model early when necessary
class EarlyStopping:
    def __init__(self, patience):
        self.patience, self.patience_counter = patience, 0
        self.best_model = None
        self.best_loss = float('inf')

    def __call__(self, model, dataset):
        new_loss = model.evaluate(dataset)
        if new_loss < self.best_loss:
            self.best_loss = new_loss
            self.best_model = model
        else:
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                self.stop(model)

    def stop(self, model):
       raise Exception("early stopping activated")
       model.stop = True


# slowely decay the lr lineary
class InverseTimeDecay:
    def __init__(self, lr=0.001, decay=5e-6):
        self.lr, self.initial_lr = lr, lr
        self.decay = decay

    def step(self, steps):
        if self.decay:
            self.lr = self.initial_lr / (1 + self.decay * steps)
        return self.lr


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


# overall model object that holds all data of the MLP
class Model:
    def __init__(self, layers: list = [], optimizer: object = Adam(), scheduler: object = InverseTimeDecay(0.001),
                 early_stopping: bool =None, loss: object =CCE(), dataset: object = None):
        # initialize each layer and activation function for forward and backward passes
        self.steps = layers
        # set the optimizer to Adam and pass the learning and decay rates
        self.optimizer = optimizer
        # set the loss function to Categorical Cross Entropy or Log Loss
        self.loss_function = loss
        # set whether early stopping is applicable
        self.early_stopping = early_stopping
        self.stop = False
        # set the scheduler
        self.scheduler = scheduler
        # set dataset
        self.dataset = dataset

    def add_layer(self, new_layer : object):
       self.steps.append(new_layer)

    def change_optimizer(self, new_optimizer : object):
        self.optimizer = new_optimizer

    def change_dataset(self, new_dataset : object):
        self.dataset = new_dataset

    def train(self, epochs : int, batch_size : int):
        for epoch in range(epochs):
            self.dataset.shuffle()
            # Calculate how many iterations to go through in one epoch
            for i in range(math.ceil(self.dataset.size[0] / batch_size)):
                # Create a batch of data and its corresponding truth values
                x_train, y_train = self.dataset.get_batch(batch_size, i)

                # Forward pass: Run through the network
                outputs = self._pass(self.steps, x_train)
                # Calculate loss gradients
                dinputs = self.loss_function.backward(outputs, y_train)

                # Backward pass of the network
                self._pass(self.steps[::-1],  dinputs, pass_type='backward')

                # Update weights and biases using the optimizer
                self._update()
                print(f'Epoch: {epoch + 1} {self.loss_function.forward(self, outputs, y_train)} Accuracy: '
                    f'{self._accuracy(outputs, y_train)} lr: {self.scheduler.lr}')

            #check if early stopping is enabled
            if self.early_stopping:
                self.early_stopping(self, self.dataset)
                if self.stop:
                    break

    def _update(self):
        # iterate through the steps, of the class is a dense layer then update the weights * biases using Adam
        for i in self.steps:
            if isinstance(i, DenseLayer) or isinstance(i, ConvolutionLayer):
                self.optimizer.update(i, self.scheduler.step(self.optimizer.t))

    def validate(self):
        # get the validation dataset and set it's truth values
        x, y_true = self.dataset.validate()
        # forward propagation inorder to determine what the ANN thinks
        self._pass(self.steps, x, False)
        return self.loss_function.forward(self, y_true)

    def test(self, batch_size):

        results = np.array([])

        for i in range(math.ceil(self.dataset.size[0] / batch_size)):
            # Create a batch of data and its corresponding truth values
            x_test, y_test = self.dataset.get_test_batch(batch_size, i)

            output = self._pass(self.steps,  x_test, False)
            print(f'Test Accuracy: {self._accuracy(output, y_test)}')


    def _pass(self, steps: list, x_batch: np.array, training: bool = False, pass_type: str = 'forward'):
        # set the X_batch to inputs inorder to loop through each layer
        inputs = x_batch
        # call each step's forward method and set it's output to inputs
        for step in steps:
            # turns off Dropout Layers when on testing set or validation set
            if training or not isinstance(step, DropoutLayer) and not training:
                inputs = getattr(step, pass_type)(inputs)
        return inputs

    @staticmethod
    def _accuracy(y_pred, y_true):
        y_pred_indices = np.argmax(y_pred, axis=1)
        y_true_indices = np.argmax(y_true, axis=1)

        return np.mean(y_pred_indices == y_true_indices)

    def save_model(self, filename:str = 'model.pkl'):
        # save the model to a certain file
        with open(filename, 'wb') as file:
            pkl.dump(self, file)