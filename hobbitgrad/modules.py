import numpy as np

class RELU:
    def __init__(self):
        self.cache = None
        
    def forward(self, X):
        self.cache = X
        return np.maximum(X, 0)
    
    def backward(self, d_out):
        X = self.cache
        dX = d_out * (X > 0)
        return dX
    
    def __call__(self, X):
        return self.forward(X)


class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size).astype(np.float32)
        self.b = np.zeros((output_size, ), dtype=np.float32)
        self.cache = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.cache = X
        # print("X.shape, self.W.shape, self.b.shape", X.shape, self.W.shape, self.b.shape)
        return X @ self.W + self.b
    
    def backward(self, d_out):
        # d_out shape: (batch_size, num_neurons)
        X = self.cache
        # Gradients for weights and bias
        self.dW = X.T @ d_out
        self.db = np.sum(d_out, axis=0)
        # Gradient to pass backward
        dX = d_out @ self.W.T
        return dX
    
    def __call__(self, X):
        return self.forward(X)

    def __repr__(self):
        return f"\n  Linear(num_neurons={len(self.neurons)}, neurons={self.neurons})\n"


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out
    
    def __call__(self, X):
        return self.forward(X)
    
    def __repr__(self):
        input_size = self.layers[0].neurons[0].W.shape[0]
        output_size = len(self.layers[-1].neurons)
        return f"Model(inputs={input_size}, num_layers={len(self.layers)}, outputs={output_size}, layers={self.layers})"

...
...

class MSELoss:
    def __init__(self):
        self.cache = None
        
    def forward(self, y, y_pred):
        self.cache = y, y_pred
        return np.mean((y - y_pred)**2)
    
    def backward(self):
        y, y_pred = self.cache
        N = y.size
        return (2.0 / N) * (y_pred - y)


class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def zero_grad(self):
        for layer in self.model.layers:
            if isinstance(layer, Linear):
                layer.dW = None
                layer.db = None