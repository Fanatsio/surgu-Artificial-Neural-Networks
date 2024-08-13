import numpy as np
from sklearn.metrics import mean_squared_error

class Neuron:
    def __init__(self, num_inputs: int):
        """
        Инициализация нейрона с заданным количеством входов.

        Параметры:
        num_inputs (int): Количество входов для нейрона.
        """
        if num_inputs <= 0:
            raise ValueError("Количество входов должно быть положительным числом.")
        self.weights = np.random.uniform(0.001, 0.2, size=num_inputs)

    def predict(self, inputs: list) -> float:
        """
        Вычисляет предсказание нейрона на основе входных данных.

        Параметры:
        inputs (list): Список входных данных.

        Возвращает:
        float: Скалярное произведение входов и весов нейрона.
        """
        if len(inputs) != len(self.weights):
            raise ValueError("Количество входных данных должно совпадать с количеством весов.")
        return np.dot(inputs, self.weights)


class NeuralNetwork:
    def __init__(self, num_neurons, num_inputs, learning_rate=0.0001):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.learning_rate = learning_rate

    def predict(self, inputs):
        return np.mean([neuron.predict(inputs) for neuron in self.neurons])

    def fit(self, x_train, y_train, x_test, y_test, desired_error, max_iterations):
        train_data = list(zip(x_train, y_train))
        test_data = list(zip(x_test, y_test))

        for _ in range(max_iterations):
            for inputs, target in train_data:
                inputs = np.array(inputs)  # Преобразуем входные данные в массив numpy
                prediction = self.predict(inputs)
                error = target - prediction
                for neuron in self.neurons:
                    neuron.weights += self.learning_rate * error * inputs

            train_predictions = [self.predict(inputs) for inputs, _ in train_data]
            train_mse = mean_squared_error(y_train, train_predictions)

            test_predictions = [self.predict(inputs) for inputs, _ in test_data]
            test_mse = mean_squared_error(y_test, test_predictions)

            if train_mse < desired_error:
                break

        print(f"Mean square error (train) -> {train_mse}")
        print(f"Mean square error (test) -> {test_mse}")

        return [neuron.weights for neuron in self.neurons]

    def fit1(self, x_train, y_train, x_test, y_test):
        mse_train = 0
        length_train = len(x_train)

        for i in range(length_train):
            inputs = np.array(x_train[i])  # Преобразуем входные данные в массив numpy
            y_pred = self.predict(inputs)
            mse_train += ((y_pred - y_train[i]) ** 2) / length_train
            for neuron in self.neurons:
                neuron.weights += (y_train[i] - y_pred) * self.learning_rate * inputs

        mse_test = sum(((self.predict(inputs) - output) ** 2) / len(x_test) for inputs, output in zip(x_test, y_test))

        print(f"Mean square error (train) -> {mse_train}")
        print(f"Mean square error (test)  -> {mse_test}")

        return [neuron.weights for neuron in self.neurons]

    def fit2(self, x_train, y_train, x_test, y_test):
        mse_train = 0
        length_train = len(x_train)

        for i in range(length_train):
            inputs = np.array(x_train[i])  # Преобразуем входные данные в массив numpy
            y_pred = self.predict(inputs)
            mse_train += ((y_pred - y_train[i]) ** 2) / length_train
            for neuron in self.neurons:
                neuron.weights -= self.learning_rate * (y_pred - y_train[i]) * inputs

        mse_test = sum(((self.predict(inputs) - output) ** 2) / len(x_test) for inputs, output in zip(x_test, y_test))

        print(f"Mean square error (train) -> {mse_train}")
        print(f"Mean square error (test)  -> {mse_test}")

        return [neuron.weights for neuron in self.neurons]


# Чтение данных из файла CSV
data_file = 'data.csv'
data = []
with open(data_file, 'r') as file:
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

# Создание и обучение нейронной сети
num_neurons = 1
num_inputs = len(x_train[0])
nn = NeuralNetwork(num_neurons=num_neurons, num_inputs=num_inputs)
trained_weights = nn.fit(x_train, y_train, x_test, y_test, desired_error=0.01, max_iterations=1000)
print("Trained weights (fit):", trained_weights)

print("------------")
trained_weights_fit1 = nn.fit1(x_train, y_train, x_test, y_test)
print("Trained weights (fit1):", trained_weights_fit1)

print("------------")
trained_weights_fit2 = nn.fit2(x_train, y_train, x_test, y_test)
print("Trained weights (fit2):", trained_weights_fit2)
