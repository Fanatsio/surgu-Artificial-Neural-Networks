from typing import List
import numpy as np
from math import sqrt

class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = np.random.uniform(0.001, 0.2, num_inputs)

    def predict(self, x: list) -> float:
        return np.dot(x, self.weights)

    def update_weights(self, x: list, y: float):
        error = y - self.predict(x)
        
        for i in range(len(self.weights)):
            self.weights[i] += error / np.sum(self.weights)

class NeuralNetwork:
    def __init__(self, num_neurons: int):
        self.neurons: List[Neuron] = [Neuron(2) for _ in range(num_neurons)]

    def predict(self, inputs: List[float]) -> list:
        return [neuron.predict(inputs) for neuron in self.neurons]

    def fit(self, x_train: list, y_train: list):
        train_mse = 1e6
        while train_mse >= 173687:
            for x, y in zip(x_train, y_train):
                for neuron in self.neurons:
                    neuron.update_weights(x, y)

            predictions = [self.predict(x) for x in x_train]
            mse = []
            for i in range(len(y_train)):
                mse.append((y_train[i] - predictions[i][0]) / 2)
            train_mse = np.mean(mse)
            print(train_mse)

data = []
with open('data.csv', 'r') as file:
    for line in file:
        if not line.startswith('x1'):
            values = line.strip().split(',')
            x = [int(values[0]), int(values[1])]
            y = float(values[2])
            data.append((x, y))

split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

x_train, y_train = zip(*train_data)
x_test, y_test = zip(*test_data)

neural_network_1 = NeuralNetwork(num_neurons=1)
neural_network_1.fit(x_train, y_train)

predictions = [neural_network_1.predict(x) for x in x_test]

mse = []
for i in range(len(y_test)):
    mse.append((y_test[i] - predictions[i][0]) / 2)
test_mse = np.mean(mse)

print(f'Test Mean Squared Error: {test_mse}')
print(f'Error: {sqrt(np.abs(test_mse))}')
