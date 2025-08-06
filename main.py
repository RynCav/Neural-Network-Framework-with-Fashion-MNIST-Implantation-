from model import *
from params import *
import pickle


# Dataset used:
Dataset = d.Data('Datasets/Fashion MNIST/')
Dataset.normalize()

# innilize each layer and activation function
STEPS = [ConvolutionLayer(32, 3, 3, 1, 1, 1), BatchNormalization(), ELU(),
             PoolingLayer(2, 2, 2),

             ConvolutionLayer(64, 3, 3, 32, 1, 1), BatchNormalization(), ELU(),
             PoolingLayer(2, 2, 2),
             FlattenLayer(),
             DenseLayer(3136, 1000), BatchNormalization(), ELU(), DropoutLayer(),
             DenseLayer(1000, 500), BatchNormalization(), ELU(), DropoutLayer(),
             DenseLayer(500, 10), Softmax()]
# create the model object
Model = Model(STEPS, dataset=Dataset)

''' # train the model
    Model.train(EPOCHS, BATCH_SIZE)

    # save the model to the specified file1
    Model.save_model()
'''

with open('model.pkl', 'rb') as file:
    Model = pickle.load(file)

Model.test(2)


'''if __name__ == '__main__':'''





