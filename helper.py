import numpy as np
from numba import cuda, jit


@cuda.jit
def convolve(outputs, weights, inputs, stride):
    i = cuda.blockIdx.z // weights.shape[0]
    k = cuda.blockIdx.z % weights.shape[0]
    h = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    w = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    H_out = outputs.shape[2]
    W_out = outputs.shape[3]

    if h >= H_out or w >= W_out:
        return

    summed = 0.0
    for c in range(inputs.shape[1]):
            for x_pos in range(weights.shape[2]):
                for y_pos in range(weights.shape[3]):

                    h_pos = h * stride + x_pos
                    w_pos = w * stride + y_pos

                    if h_pos < inputs.shape[2] and w_pos < inputs.shape[3]:
                        summed += inputs[i, c, h_pos, w_pos] * weights[k, c, x_pos, y_pos]

    outputs[i, k, h, w] = summed

@jit
def dconvolve(dweights, weights, dinputs, dvalues, inputs, stride):
    for i in range(dvalues.shape[0]):
        for k, kernel in enumerate(weights):
            for h in range(dvalues.shape[2]):
                for w in range(dvalues.shape[3]):
                    h_start = h * stride
                    w_start = w * stride

                    h_end = h_start + weights.shape[2]
                    w_end = w_start + weights.shape[3]

                    dweights[k] += dvalues[i, k, h, w] * inputs[i, :, h_start:h_end, w_start:w_end]
                    dinputs[i, :, h_start:h_end, w_start:w_end] += dvalues[i, k, h, w] * kernel
    return dweights, dinputs