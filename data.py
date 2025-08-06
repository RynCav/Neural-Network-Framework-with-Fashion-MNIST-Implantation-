import numpy as np

# inherited class, template
class Data:
    # save file path and if validation set present
    def __init__(self, directory):
        # load data
        self.train_X = np.load(f'{directory}Train_X.npy').astype('float32')
        self.test_X  = np.load(f'{directory}Test_X.npy').astype('float32')

        # load truth values
        self.train_y = np.load(f'{directory}Train_y.npy').astype('float32')
        self.test_y = np.load(f'{directory}Test_y.npy').astype('float32')

        # load the validation set if applicable
        try:
            self.val_X = np.load(f'{directory}Val_X.npy').astype('float32')
            self.val_y = np.load(f'{directory}Val_X.npy').astype('float32')
        except:
           print('Validation set not found')

        # get the size of the dataset
        self.size = self.train_X.shape

        #save if there is a validation set
        self.val = hasattr(self, 'val_X')

    # normilize across all data for images between -1 - 1.
    def normalize(self):
        scale = np.max(self.train_X) / 2
        self.train_X = (self.train_X - scale) / scale
        self.test_X = (self.test_X - scale) / scale

        try:
            self.val_X = (self.val_X - scale) / scale
        except:
            print('Validation set not normalized because it doesnt exist')

    # randomly shuffle data for each epoch in training
    def shuffle(self):
        shuffled_indices = np.random.permutation(len(self.train_X))
        self.train_X, self.train_y = self.train_X[shuffled_indices], self.train_y[shuffled_indices]

    # get a batch of the training set
    def get_batch(self, batch_size, i):
        # returns a batch of n size that is conatins data that hasnt been trained before in the current Epoch
        return self.train_X[i * batch_size:(i + 1) * batch_size], self.train_y[i * batch_size:(i + 1) * batch_size]

    # get a batch of the training set
    def get_test_batch(self, batch_size, i):
        # returns a batch of n size that is conatins data that hasnt been trained before in the current Epoch
        return self.test_X[i * batch_size:(i + 1) * batch_size], self.test_y[i * batch_size:(i + 1) * batch_size]

    # returns validation set
    def validation(self):
        try:
            return self.val_X, self.val_y
        except:
            raise Exception('Validation set not found')

    # load the testing data
    def test(self):
        # returns the testing data
        return self.test_X, self.test_y

    # save data
    def save(self, directory = 'Datasets/'):
        np.save(f'{directory}Train_X.npy', self.train_X)
        np.save(f'{directory}Test_X.npy', self.test_X)

        np.save(f'{directory}Train_y.npy', self.train_y)
        np.save(f'{directory}Test_y.npy', self.test_y)

        try:
            np.save(f'{directory}Val_X.npy', self.val_X)
            np.save(f'{directory}Val_y.npy', self.val_y)
        except:
            print('Validation set not saved because it doesnt exist')
