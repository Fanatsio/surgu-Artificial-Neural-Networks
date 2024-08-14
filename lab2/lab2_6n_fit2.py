import numpy as np
from typing import List
from math import sqrt

class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = np.random.uniform(0.001, 0.2, num_inputs)
        self.learning_rate = 1e-4

    def predict(self, x: list) -> float:
        return np.dot(x, self.weights)
    
    def update_weights(self, x: list, y: float):
        prediction = self.predict(x)
        error = prediction - y
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * error * x[i]

class NeuralNetwork:
    def __init__(self, num_neurons: int, num_inputs: int):
        self.neurons: List[Neuron] = [Neuron(num_inputs) for _ in range(num_neurons)]

    def predict(self, inputs: List[float]) -> list:
        return [neuron.predict(inputs) for neuron in self.neurons]

    def fit(self, x_train: list, y_train: list):
        train_mse = 10
        while np.abs(train_mse) > 2:
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
            x = [int(v) for v in values[:6]]
            y = [float(v) for v in values[6:]]
            data.append((x, y))

split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

x_train, y_train = zip(*train_data)
x_test, y_test = zip(*test_data)

neural_network = NeuralNetwork(num_neurons=3, num_inputs=6)
neural_network.fit(x_train, y_train)

predictions = [neural_network.predict(x) for x in x_test]

print(f"\n{y_test}\n")
print(f"{predictions}\n")

mse = []
for i in range(len(y_test)):
    mse.append(np.mean([(y_test[i][j] - predictions[i][j]) / 2 for j in range(len(neural_network.neurons))]))
test_mse = np.mean(mse)
    
print(f'Test Mean Squared Error: {test_mse}')
print(f'Error: {sqrt(test_mse)}')
