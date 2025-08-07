import data as d

# Hyperparameters
BATCH_SIZE = 100
EPOCHS = 10
lr = 0.0001
max_lr = 0.015
dr = 0.7
DECAY = 1e-6
PATIENCE = 3

# Dataset used:
Dataset = d.Data('Datasets/Fashion MNIST/')
Dataset.normalize()
