import numpy as np


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
