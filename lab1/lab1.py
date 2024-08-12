import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.training_inputs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        self.labels = np.array([0, 0, 1, 0])
        self.weights = np.random.rand(self.training_inputs.shape[1] + 1)  # Добавляем смещение
        self.learning_rate = learning_rate

        print(f"Начальные веса = {self.weights[1:]}, Смещение = {self.weights[0]}")

    def predict(self, inputs):
        # Учитываем смещение при вычислении суммы
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation >= 0 else 0

    def train_step(self):
        total_error = 0
        for inputs, label in zip(self.training_inputs, self.labels):
            prediction = self.predict(inputs)
            error = label - prediction
            total_error += abs(error)
            # Обновление весов и смещения
            self.weights[1:] += self.learning_rate * error * inputs
            self.weights[0] += self.learning_rate * error
            print(f"Веса во время обучения {self.weights[1:]}, Смещение = {self.weights[0]}")
        return total_error

    def test(self) -> bool:
        correct_predictions = all(self.predict(inputs) == label for inputs, label in zip(self.training_inputs, self.labels))
        return correct_predictions

perceptron = Perceptron()

epoch = 0
while not perceptron.test():
    total_error = perceptron.train_step()
    epoch += 1
    print(f"Эпоха {epoch}, Общая ошибка: {total_error}")

print(f"Веса после обучения {perceptron.weights[1:]}, Смещение = {perceptron.weights[0]}")

print("Тестирование модели:")
for inputs in perceptron.training_inputs:
    prediction = perceptron.predict(inputs)
    print(f"Вход: {inputs}, Предсказание: {prediction}")
