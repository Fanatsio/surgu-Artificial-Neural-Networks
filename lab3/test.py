import numpy as np


class Neuron:
    def __init__(self, weights, activation='sigmoid'):
        # Инициализация весов для нейрона
        self.weights = np.array(weights, dtype=float)  # Начальные веса из задания
        self.activation = activation
        self.output = 0
        self.delta = 0

    def predict(self, x):
        # Прямой проход: скалярное произведение + смещение
        return np.dot(x, self.weights)

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, output):
        # Производная сигмоиды
        return output * (1 - output)

    def update_weights(self, x, learning_rate):
        # Обновление весов
        self.weights += learning_rate * self.delta * np.array(x)


class NeuralNetwork:
    def __init__(self, init_weights_randomly=True):
        # Если init_weights_randomly=True, инициализируем веса случайными малыми значениями
        if init_weights_randomly:
            self.hidden_layer = [
                Neuron(np.random.uniform(-0.5, 0.5, 3)),
                Neuron(np.random.uniform(-0.5, 0.5, 3)),
                Neuron(np.random.uniform(-0.5, 0.5, 3))
            ]
            self.output_layer = [
                Neuron(np.random.uniform(-0.5, 0.5, 3)),
                Neuron(np.random.uniform(-0.5, 0.5, 3))
            ]
        else:
            # Использование фиксированных весов из задания
            self.hidden_layer = [
                Neuron([1, 4, -3]),
                Neuron([5, -2, 4]),
                Neuron([2, -3, 1])
            ]
            self.output_layer = [
                Neuron([2, 4, -2]),
                Neuron([-3, 2, 3])
            ]

    def feedforward(self, x):
        # Прямой проход
        hidden_outputs = [neuron.activate(neuron.predict(x)) for neuron in self.hidden_layer]
        final_outputs = [neuron.activate(neuron.predict(hidden_outputs)) for neuron in self.output_layer]
        return hidden_outputs, final_outputs

    def backpropagation(self, x, y, learning_rate):
        # Прямой проход для предсказания
        hidden_outputs, final_outputs = self.feedforward(x)

        hidden_outputs, final_outputs = self.feedforward(x)
        for i, neuron in enumerate(self.output_layer):
            error = y[i] - final_outputs[i]
            neuron.delta = error * neuron.sigmoid_derivative(final_outputs[i])

        # Обновление скрытого слоя, обобщённое дельта-правило:
        for i, neuron in enumerate(self.hidden_layer):
            error = sum(output_neuron.delta * output_neuron.weights[i] for output_neuron in self.output_layer)
            neuron.delta = error * neuron.sigmoid_derivative(hidden_outputs[i])

        # Обновление весов выходного слоя
        for neuron in self.output_layer:
            neuron.update_weights(hidden_outputs, learning_rate)

        # Обновление весов скрытого слоя
        for neuron in self.hidden_layer:
            neuron.update_weights(x, learning_rate)

    def fit(self, X, y, learning_rate=0.01, tolerance=1e-6, max_epochs=1000):
        for epoch in range(max_epochs):
            total_error = 0
            for i in range(len(X)):
                self.backpropagation(X[i], y[i], learning_rate)
                outputs = self.predict(X[i])
                errors = [(y[i][j] - outputs[j]) for j in range(len(outputs))]
                total_error += sum(error ** 2 for error in errors)
                print(
                    f"i = {i}, Epoch {epoch + 1}, Error {errors},"
                    f"  Предикт {outputs}, Таргет {y[i]}")

            mse = total_error / (len(X) * len(y[0]))
            print(f"Epoch {epoch + 1}, MSE: {mse}")

            if mse < tolerance:
                print("Training stopped due to tolerance level.")
                break

    def predict(self, X):
        # Предсказание
        _, final_outputs = self.feedforward(X)
        return final_outputs


# Пример использования
import pandas as pd

# Загрузка данных
data = pd.read_csv('data/3lab_data.csv')
X = data[['x1', 'x2', 'x3']].values  # Входные данные
y = data[['y1', 'y2']].values

# Нормализация данных
X_max = np.max(X, axis=0)
y_max = np.max(y, axis=0)
X_normalized = X / X_max
y_normalized = y / y_max

# Инициализация сети
network = NeuralNetwork()

# Обучение сети
network = NeuralNetwork(init_weights_randomly=True)
network.fit(X_normalized, y_normalized, learning_rate=0.01, tolerance=1e-6, max_epochs=1000)

