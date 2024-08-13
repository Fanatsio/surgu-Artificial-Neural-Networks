from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error


class Neuron:
    def __init__(self, num_inputs: int):
        if num_inputs <= 0:
            raise ValueError("Количество входов должно быть положительным числом.")
        self.weights = np.random.uniform(0.001, 0.2, size=num_inputs)

    def predict(self, inputs: list) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError("Количество входных данных должно совпадать с количеством весов.")
        return np.dot(inputs, self.weights)


class NeuralNetwork:
    def __init__(self, num_neurons: int, num_inputs: int, learning_rate: float = 0.0001) -> None:
        self.neurons: List[Neuron] = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.learning_rate: float = learning_rate

    def predict(self, inputs: List[float]) -> float:
        if len(inputs) != len(self.neurons[0].weights):
            raise ValueError("Количество входных данных должно совпадать с количеством весов в нейронах.")
        predictions = [neuron.predict(inputs) for neuron in self.neurons]
        return np.mean(predictions)

    def fit(self, x_train, y_train, x_test, y_test, desired_error, max_iterations):
        if not all(len(inputs) == len(self.neurons[0].weights) for inputs in x_train):
            raise ValueError("Все входные данные должны иметь ту же длину, что и количество весов в нейронах.")

        train_data = list(zip(x_train, y_train))
        test_data = list(zip(x_test, y_test))

        for _ in range(max_iterations):
            for inputs, target in train_data:
                inputs = np.array(inputs)  # Преобразуем входные данные в массив numpy
                prediction = self.predict(inputs)
                error = target - prediction
                for neuron in self.neurons:
                    weight_sum = np.sum(neuron.weights)  # Sum of the weights
                    neuron.weights += (self.learning_rate * error * inputs) / weight_sum


            train_predictions = [self.predict(inputs) for inputs, _ in train_data]
            train_mse = mean_squared_error(y_train, train_predictions)

            test_predictions = [self.predict(inputs) for inputs, _ in test_data]
            test_mse = mean_squared_error(y_test, test_predictions)

            if train_mse < desired_error:
                break

        print(f"Mean square error (train) -> {train_mse}")
        print(f"Mean square error (test) -> {test_mse}")

        return [neuron.weights for neuron in self.neurons]

    def fit1(self, x_train: List[List[float]], y_train: List[float], x_test: List[List[float]], y_test: List[float]):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        mse_train = 0
        length_train = len(x_train)

        # Суммирование ошибок
        for i in range(length_train):
            inputs = x_train[i]
            y_pred = self.predict(inputs)
            mse_train += ((y_pred - y_train[i]) ** 2) / length_train
            error = y_train[i] - y_pred
            for j, neuron in enumerate(self.neurons):
                neuron.weights += self.learning_rate * error * inputs[j]

        y_test_pred = np.array([self.predict(inputs) for inputs in x_test])
        mse_test = mean_squared_error(y_test, y_test_pred)

        print(f"Mean square error (train) -> {mse_train}")
        print(f"Mean square error (test)  -> {mse_test}")

        return [neuron.weights for neuron in self.neurons]