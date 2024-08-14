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
        self.neurons: List[Neuron] = [Neuron(6) for _ in range(num_neurons)]

    def predict(self, inputs: List[float]) -> list:
        return [neuron.predict(inputs) for neuron in self.neurons]

    def fit(self, x_train: list, y_train: list):
        train_mse = 1e7
        # while train_mse >= 173687:
        while train_mse >= 1e6:
            for x, y in zip(x_train, y_train):
                for i, neuron in enumerate(self.neurons):
                    neuron.update_weights(x, y[i])

            predictions = [self.predict(x) for x in x_train]
            mse = []
            for i in range(len(y_train)):
                mse.append(np.mean([(y_train[i][j] - predictions[i][j]) / 2 for j in range(len(self.neurons))]))
            train_mse = np.mean(mse)
            print(train_mse)

data = []
with open('2lab_data.csv', 'r') as file:
    for line in file:
        if not line.startswith('x1'):
            values = line.strip().split(',')
            x = [int(values[j]) for j in range(6)]
            y = [float(values[j]) for j in range(6, 9)]
            data.append((x, y))

split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

x_train, y_train = zip(*train_data)
x_test, y_test = zip(*test_data)

neural_network_3 = NeuralNetwork(num_neurons=3)
neural_network_3.fit(x_train, y_train)

predictions = [neural_network_3.predict(x) for x in x_test]

mse = []
for i in range(len(y_test)):
    mse.append(np.mean([(y_test[i][j] - predictions[i][j]) / 2 for j in range(3)]))
test_mse = np.mean(mse)

print(f'Test Mean Squared Error: {test_mse}')
print(f'Error: {sqrt(np.abs(test_mse))}')
