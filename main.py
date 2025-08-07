from model import *
from params import *
import pickle


if __name__ == '__main__':
    # innilize each layer and activation function
    STEPS = [nnf.ConvolutionLayer(32, 3, 3, 1, 1, 1), nnf.BatchNormalization(), nnf.ELU(),
                 nnf.PoolingLayer(2, 2, 2),
    
                 nnf.ConvolutionLayer(64, 3, 3, 32, 1, 1), nnf.BatchNormalization(), nnf.ELU(),
                 nnf.PoolingLayer(2, 2, 2),
                 nnf.FlattenLayer(),
                 nnf.DenseLayer(3136, 1000), nnf.BatchNormalization(), nnf.ELU(), nnf.DropoutLayer(),
                 nnf.DenseLayer(1000, 500), nnf.BatchNormalization(), nnf.ELU(), nnf.DropoutLayer(),
                 nnf.DenseLayer(500, 10), nnf.Softmax()]
    # create the model object
    Model = Model(STEPS, dataset=Dataset)
    
     # train the model
    Model.train(EPOCHS, BATCH_SIZE)
    
    # save the model to the specified file1
    Model.save_model()



