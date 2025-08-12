import numpy as np

np.random.seed(42)

def relu(x):
    return np.maximum(x, 0)

class Neuron:
    def __init__(self, input_size):
        self.w = np.random.rand(input_size).astype(np.float32)
        self.b = np.zeros(1, dtype=np.float32)

    def forward(self, x):
        # print("    x.shape", x.shape)
        neuron_out = x @ self.w + self.b
        # print("    neuron_out.shape, neuron_out:", neuron_out.shape, neuron_out)
        return neuron_out
    
    def __repr__(self):
        return f"\n    Neuron(w.shape={self.w.shape}, b.shape={self.b.shape}, dtype={self.w.dtype})"

class Layer:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for i in range(output_size)]

    def forward(self, x):
        layer_outputs = [neuron.forward(x) for neuron in self.neurons]
        return np.column_stack(layer_outputs)
    
    def __repr__(self):
        return f"\n  Layer(num_neurons={len(self.neurons)}, neurons={self.neurons})\n"


class Model:
    def __init__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

    def forward(self, x):
        for i in range(len(self.layers)):
            # print(f"layer {i+1}")
            x = self.layers[i].forward(x)
        return x
    
    def __repr__(self):
        input_size = self.layers[0].neurons[0].w.shape[0]
        output_size = len(self.layers[-1].neurons)
        return f"Model(inputs={input_size}, num_layers={len(self.layers)}, outputs={output_size}, layers={self.layers})"


def mse(y, ypred):
    mse = np.mean((y - ypred)**2)
    return mse

m = 10
features = 3

x = np.random.rand(m, features)

true_weights = np.array([3.0, -1.0, 6.5])
true_bias = 3.0

y = x @ true_weights + true_bias + np.random.randn(m) * 0.1
y = y.reshape(-1, 1)
print("y:\n", y)

layer_sizes = [features, 2 * features, features, 1]
model = Model(layer_sizes)
# print(model)

ypred = model.forward(x)
print("ypred:\n", ypred)

mse_loss = mse(y, ypred)
print("mse loss:", mse_loss)

# backward = model.backward()
