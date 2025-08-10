import numpy as np

np.random.seed(42)

def relu(x):
    return np.maximum(x, 0)

class Layer:
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        print("self.W.shape, self.b.shape:", self.W.shape, self.b.shape)

    def forward(self, x):
        self.x = x
        print("x.shape", x.shape)
        print("self.W.shape, self.b.shape:", self.W.shape, self.b.shape)
        print()
        return x @ self.W + self.b
    
    def backward(self, dout):
        ...

class Model:
    def __init__(self, layers):
        layers.append(1)
        self.layers = [Layer(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        print("len(self.layers):", len(self.layers))

    def forward(self, data):
        for layer in self.layers:
            print("data.shape:",data.shape)
            data = layer.forward(data)
        print("data.shape:",data.shape)
        print(data)       

m = 10
input_dim = 3

X = np.random.rand(m, input_dim)
print("X.shape:", X.shape)

true_weights = np.array([3.0, -1.0, 6.5])
true_bias = 3.0

y = X @ true_weights + true_bias + np.random.randn(m) * 0.1
y = y.reshape(-1, 1)
print("y.shape:", y.shape)

layers = [input_dim, 2 * input_dim, input_dim]
model = Model(layers)
model.forward(X)
