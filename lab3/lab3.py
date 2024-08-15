import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = np.random.uniform(0.001, 0.2, num_inputs)
        self.learning_rate = 1e-5

    def predict(self, x: np.ndarray) -> float:
        return sigmoid(np.dot(x, self.weights))
    
    def update_weights(self, x: np.ndarray, error: float):
        delta = error * sigmoid_derivative(self.predict(x))
        self.weights += self.learning_rate * delta * x

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация слоёв сети
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        hidden_outputs = np.array([neuron.predict(inputs) for neuron in self.hidden_layer])
        final_outputs = np.array([neuron.predict(hidden_outputs) for neuron in self.output_layer])
        return final_outputs

    def backpropagate(self, inputs: np.ndarray, y: np.ndarray):
        # Прямое распространение
        hidden_outputs = np.array([neuron.predict(inputs) for neuron in self.hidden_layer])
        final_outputs = np.array([neuron.predict(hidden_outputs) for neuron in self.output_layer])
        
        # Вычисление ошибки для выходного слоя
        output_errors = y - final_outputs
        
        # Обновление весов выходного слоя
        for i, neuron in enumerate(self.output_layer):
            neuron.update_weights(hidden_outputs, output_errors[i])
        
        # Вычисление ошибки для скрытого слоя
        hidden_errors = np.dot(output_errors, np.array([neuron.weights for neuron in self.output_layer]))
        
        # Обновление весов скрытого слоя
        for i, neuron in enumerate(self.hidden_layer):
            neuron.update_weights(inputs, hidden_errors[i])

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs=1000):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                self.backpropagate(np.array(x), np.array(y))
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch} Weights:')
                self.print_weights()

    def print_weights(self):
        print("Hidden Layer Weights:")
        for i, neuron in enumerate(self.hidden_layer):
            print(f"Neuron {i+1}: {neuron.weights}")
        print("Output Layer Weights:")
        for i, neuron in enumerate(self.output_layer):
            print(f"Neuron {i+1}: {neuron.weights}")

# Загрузка данных из файла 3lab_data.csv
data = np.genfromtxt('3lab_data.csv', delimiter=',', skip_header=1)
x_data = data[:, :3]
y_data = data[:, 3:]

# Разделение на тренировочные и тестовые данные
split_index = int(0.8 * len(data))
x_train, x_test = x_data[:split_index], x_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

# Создание и обучение сети
nn = NeuralNetwork(input_size=3, hidden_size=3, output_size=2)
nn.fit(x_train, y_train)

# Тестирование
print("Final Weights After Training:")
nn.print_weights()

predictions = np.array([nn.predict(x) for x in x_test])
