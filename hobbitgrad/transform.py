import random
from math import exp

random.seed(42)

def relu(x):
    return max(x, 0)
 
def init_param(shape):
    # initialize a parameter
    return random.random()

def init_neuron(w_shape, b_shape):
    # initialize a neuron 
    return (init_param(w_shape), init_param(b_shape))

def init_layer(num_neurons, num_prev_layer_neurons, num_next_layer_neurons):
    # initialize a layer
    layer = []
    for i in range(num_neurons):
        layer.append(init_neuron(num_prev_layer_neurons))
    return layer

def linear_forward(W, X, b):
    Z = np.matmul(W, X) + b
    # cache = (W, X, b) # implement this where this is called? because why create it here at all if just for this
    return Z

class Model:
    def __init__(self, data, layers, labels):
        param_sizes = [[len(data[0]), layers[0]]]
        for i in range(len(layers) - 1):
            param_sizes.append([layers[i], layers[i+1]])
        param_sizes.append([layers[-1], len(labels)])
        print(param_sizes)

data = [list(i for i in range(5)) for i in range(10)]
labels = [i for i in range(10)]
m = len(data)
print(data)
layers = [2 * m, 3 * m, 2 * m]
model = Model(data=data, layers=layers, labels=labels)
