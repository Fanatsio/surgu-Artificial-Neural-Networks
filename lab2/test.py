import random

#random.randint(1, 200) / 1000 for _ in range(n)
class Neuron:
    def __init__(self, n):
        self.weights = [random.randint(1, 200) / 1000 for _ in range(n)]

    def predict(self, x: list) -> float:
        return x[0] * self.weights[0] + x[1] * self.weights[1]


class NeuralNetwork:
    def __init__(self, n):
        self.neurons = [Neuron(2) for _ in range(n)]

    def predict(self, x: list):
        return self.neurons[0].predict(x)

    def fit(self, x: list, y: list):
        mse = 0
        length = len(x)
        for i in range(length):
            y_pred = self.predict(x[i])
            mse += ((y_pred - y[i]) ** 2) / length
            self.neurons[0].weights[0] += (y[i] - y_pred) / sum(self.neurons[0].weights)
            self.neurons[0].weights[1] += (y[i] - y_pred) / sum(self.neurons[0].weights)
            print(self.neurons[0].weights[0])
        print(mse)
        return [self.neurons[0].weights[0], self.neurons[0].weights[1]]


# Чтение данных из файла CSV
data = []
with open('data.csv', 'r') as file:
    for line in file:
        if not line.startswith('x1'):
            values = line.strip().split(',')
            x = [int(values[0]), int(values[1])]
            y = float(values[2])
            data.append((x, y))

# Разделение данных на обучающую и тестовую выборки
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

# Разделение x и y для обучающей и тестовой выборок
x_train, y_train = zip(*train_data)
x_test, y_test = zip(*test_data)

nn = NeuralNetwork(1)
print(nn.fit(x, y))