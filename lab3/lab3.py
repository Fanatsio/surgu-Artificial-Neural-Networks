import random
import math

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.randint(1, 200) / 1000 for _ in range(num_inputs)]
        self.bias = random.randint(1, 200) / 1000
        self.output = 0
        self.delta = 0

    def sigmoid(self, x):
        return x

    def sigmoid_derivative(self, x):
        return 1

    def feed_forward(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = self.sigmoid(weighted_sum)
        return self.output

class NeuralLayer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def feed_forward(self, inputs):
        return [neuron.feed_forward(inputs) for neuron in self.neurons]

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layers = []

        prev_layer_size = input_size
        for layer_size in hidden_sizes:
            self.layers.append(NeuralLayer(layer_size, prev_layer_size))
            prev_layer_size = layer_size

        self.layers.append(NeuralLayer(output_size, prev_layer_size))

    def feed_forward(self, inputs):
        layer_output = inputs
        for layer in self.layers:
            layer_output = layer.feed_forward(layer_output)
        return layer_output

    def backpropagate(self, inputs, targets, learning_rate):
        layer_outputs = [inputs]
        for layer in self.layers:
            layer_outputs.append(layer.feed_forward(layer_outputs[-1]))

        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = targets[i] - neuron.output
            neuron.delta = error * neuron.sigmoid_derivative(neuron.output)

        for layer_index in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]

            for i, neuron in enumerate(current_layer.neurons):
                error = sum(neuron.weights[i] * next_neuron.delta for next_neuron in next_layer.neurons)
                neuron.delta = error * neuron.sigmoid_derivative(neuron.output)

        for layer_index in range(1, len(self.layers)):
            current_layer = self.layers[layer_index]
            prev_layer = self.layers[layer_index - 1]

            for i, neuron in enumerate(current_layer.neurons):
                for j, prev_neuron in enumerate(prev_layer.neurons):
                    neuron.weights[j] += learning_rate * neuron.delta * prev_neuron.output
                neuron.bias += learning_rate * neuron.delta
        for neuron in self.layers[0].neurons:
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * neuron.delta * prev_neuron.output
            neuron.bias += learning_rate * neuron.delta

    def train(self, x, y, epochs, learning_rate):
        for _ in range(epochs):
            length = len(x)
            for i in range(length):
                self.backpropagate(x[i], y[i], learning_rate)
        print([self.layers[len(self.layers) - 1].neurons[i].weights for i in range(len(self.layers[len(self.layers) - 1].neurons))])

    def calc_mse(self, x_test, y_test):
        length = len(x_test)
        mse = 0
        for i in range(length):
            y_pred = self.predict(x_test[i])
            for j in range(len(y_pred)):
                mse += ((y_pred[j] - y_test[i][j]) ** 2) / length
        print(mse)


    def predict(self, x):
        pred = x
        for layer in self.layers:
            temp = []
            for neuron in layer.neurons:
                temp.append(neuron.feed_forward(pred))
            pred = temp.copy()
        return pred


f = open('2lab_data.csv', 'r')
x = []
y = []
x_test = []
y_test = []

count = 0
for line in f.read().split('\n'):
    lst = line.split(',')
    if count % 5:
        x.append([int(lst[i]) for i in range(6)])
        y.append([int(lst[6]), int(lst[7]), int(lst[8])])
    else:
        x_test.append([int(lst[i]) for i in range(6)])
        y_test.append([int(lst[6]), int(lst[7]), int(lst[8])])
    count += 1

nn = NeuralNetwork(input_size=6, hidden_sizes=[3, 3], output_size=3)
nn.train(x, y, epochs=1000, learning_rate=0.000000005)
nn.calc_mse(x_test, y_test)

f = open('data.csv', 'r')
x = []
y = []
x_test = []
y_test = []
count = 0
for line in f.read().split('\n'):
    lst = line.split(';')
    if count % 5:
        x.append([int(lst[0]), int(lst[1])])
        y.append([int(lst[2])])
    else:
        x_test.append([int(lst[0]), int(lst[1])])
        y_test.append([int(lst[2])])
    count += 1

nn = NeuralNetwork(input_size=2, hidden_sizes=[2, 2, 2], output_size=1)
nn.train(x, y, epochs=1000, learning_rate=0.000000005)
nn.calc_mse(x_test, y_test)
