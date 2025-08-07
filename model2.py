import numpy as np
import pickle as pkl
import nn_framework as nnf
import math


# overall model object that holds all data of the MLP
class Model:
    def __init__(self, layers: list = [], optimizer: object = nnf.Adam(), scheduler: object = nnf.InverseTimeDecay(0.001),
                 early_stopping: bool =None, loss: object = nnf.CCE(), dataset: object = None):
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
            if isinstance(i, nnf.DenseLayer) or isinstance(i, nnf.ConvolutionLayer):
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
            if training or not isinstance(step, nnf.DropoutLayer) and not training:
                inputs = getattr(step, pass_type)(inputs)
        return inputs

    @staticmethod
    def _accuracy(y_pred, y_true):
        y_pred_indices = np.argmax(y_pred, axis=1)
        y_true_indices = np.argmax(y_true, axis=1)

        return np.mean(y_pred_indices == y_true_indices)

    def save_model(self, filename:str = 'test_model.pkl'):
        # save the model to a certain file
        with open(filename, 'wb') as file:
            pkl.dump(self, file)

