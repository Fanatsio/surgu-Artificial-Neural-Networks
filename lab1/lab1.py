import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = np.random.uniform(0.1, 0.9, 2)
        self.inputs = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
        self.testSet =  np.array([0, 0, 1, 0])
        self.accuracy = 1e-2
        self.bias = 1

    def train(self):
        for inputs, testSet in zip(self.inputs, self.testSet):
            error = testSet - self.predict(inputs)
            self.weights += self.accuracy * error * inputs
            self.bias += self.accuracy * error

    def test(self) -> bool:
        correct_predictions = all(self.predict(inputs) == testSet for inputs, testSet in zip(self.inputs, self.testSet))
        return correct_predictions

    def predict(self, x):
        summation = x[0] * self.weights[0] + x[1] * self.weights[1] + self.bias
        return 1 if summation >= 0 else 0

    def get_weight(self):
        return self.weights

neuron = Perceptron()
print(f"Starting weights = {neuron.get_weight()}\n")

epoch = 0
while not neuron.test():
    neuron.train()
    epoch += 1
    print(f"Epoch: {epoch}, Weights: {neuron.get_weight()}")

print(f"\nWeights after training: {neuron.get_weight()}")
