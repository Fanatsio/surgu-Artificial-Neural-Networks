from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error

class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = np.random.uniform(0.001, 0.2, num_inputs)
        self.accuracy = 0.1

    def predict(self, x: list) -> float:
        return np.dot(x, self.weights)

class NeuralNetwork:
    def __init__(self, num_neurons: int):
        self.neurons: List[Neuron] = [Neuron(2) for _ in range(num_neurons)]
        self.learning_rate: float = 1
        
    def predict(self, inputs: List[float]) -> list:
        return [neuron.predict(inputs) for neuron in self.neurons]

    def fit(self, x : list, y : list):
        train_data = list(zip(x, y))

        for _ in range(1000):
            for inputs, target in train_data:
                inputs = np.array(inputs)
                prediction = self.predict(inputs)
                error = target - prediction[0]
                for neuron in self.neurons:
                    print(prediction[0])
                    # neuron.weights += (self.learning_rate - prediction[0]) / np.sum(neuron.weights)

        predicted = self.predict(x)
        train_mse = mean_squared_error(y, predicted[0])

        print(f"Mean square error (train) -> {train_mse}")
        
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

neural_network_1 = NeuralNetwork(num_neurons = 1)
neural_network_1.fit(x_train, y_train)
