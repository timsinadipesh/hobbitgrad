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
    

class Neuron:
    def __init__(self, input_size):
        self.W = np.random.rand(input_size, 1).astype(np.float32)
        self.b = np.zeros((1,), dtype=np.float32)
        self.cache = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.cache = X
        # print("    X.shape", X.shape)
        neuron_out = X @ self.W + self.b
        # print("    neuron_out.shape, neuron_out:", neuron_out.shape, neuron_out)
        return neuron_out
    
    def backward(self, d_out):
        X = self.cache
        # Gradients for weights and bias
        self.dW = X.T @ d_out
        self.db = np.sum(d_out, axis=0)
        # Gradient to pass backward
        dX = d_out @ self.W.T
        return dX
    
    def __repr__(self):
        return f"\n    Neuron(W.shape={self.W.shape}, b.shape={self.b.shape}, dtype={self.W.dtype})"


class Dense:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for i in range(output_size)]

    def forward(self, X):
        self.cache = X
        layer_outputs = [neuron.forward(X) for neuron in self.neurons]
        return np.column_stack(layer_outputs) # output shape (n_samples, num_neurons)
    
    def backward(self, d_out):
        # d_out shape: (batch_size, num_neurons)
        grads = []
        for i, neuron in enumerate(self.neurons):
            dX_i = neuron.backward(d_out[:, [i]])
            grads.append(dX_i)
        # Sum gradients from all neurons
        dX = np.sum(grads, axis=0)
        return dX

    def __repr__(self):
        return f"\n  Dense(num_neurons={len(self.neurons)}, neurons={self.neurons})\n"


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for i in range(len(self.layers)):
            X = self.layers[i].forward(X)
        return X
    
    def backward(self, d_out):
        for i in reversed(range(len(self.layers))):
            d_out = self.layers[i].backward(d_out)
        return d_out
    
    def __repr__(self):
        input_size = self.layers[0].neurons[0].W.shape[0]
        output_size = len(self.layers[-1].neurons)
        return f"Model(inputs={input_size}, num_layers={len(self.layers)}, outputs={output_size}, layers={self.layers})"


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
            if isinstance(layer, Dense):
                for neuron in layer.neurons:
                    neuron.W -= self.lr * neuron.dW
                    neuron.b -= self.lr * neuron.db