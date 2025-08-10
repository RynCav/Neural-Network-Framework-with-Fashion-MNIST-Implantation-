import data as d

# Hyperparameters
BATCH_SIZE = 100
EPOCHS = 10
lr = 0.001
dr = 0.7
DECAY = 1e-5

# Dataset used:
Dataset = d.Data('Datasets/Fashion MNIST/')
Dataset.normalize()

