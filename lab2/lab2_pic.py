import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = 0.01

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.inputs, self.weights) + self.bias
        self.output = self.sigmoid(self.z)
        return self.output

    def backward(self, error):
        self.error = error
        self.dz = error * self.sigmoid_derivative(self.z)
        self.dweights = np.dot(self.inputs.T, self.dz)
        self.dbias = np.sum(self.dz)
        self.weights -= self.learning_rate * self.dweights
        self.bias -= self.learning_rate * self.dbias

class NeuralNetwork:
    def __init__(self, input_size):
        self.neuron = Neuron(input_size)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                inputs = X[i].reshape(1, -1)
                target = y[i]
                prediction = self.neuron.forward(inputs)
                error = target - prediction
                self.neuron.backward(error)

            if epoch % 100 == 0:
                loss = np.mean((y - self.predict(X)) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        predictions = np.array([self.neuron.forward(x.reshape(1, -1)) for x in X])
        return predictions

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    return img_array

def get_unique_colors(img_array):
    img_array = img_array[::10, ::10]  # Уменьшение изображения
    unique_colors = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)
    return unique_colors

def preprocess_colors(colors):
    return colors / 255.0  # Нормализация

def plot_unique_colors(colors):
    # Преобразуем цвета обратно в диапазон [0, 1] для корректной визуализации
    colors = colors / 255.0

    # Создаем изображение из уникальных цветов
    color_img = np.zeros((50, len(colors), 3))
    for i, color in enumerate(colors):
        color_img[:, i, :] = color

    plt.imshow(color_img)
    plt.title("Unique Colors in Image")
    plt.axis('off')
    plt.show()

def plot_loss(epoch_loss):
    plt.plot(epoch_loss)
    plt.title('Loss during training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def main():
    img_path = 'kotik.jpg'  # Замените на путь к вашему изображению
    img_array = load_image(img_path)
    
    unique_colors = get_unique_colors(img_array)
    
    print("Unique colors found in the image (RGB format):")
    for color in unique_colors:
        print(f"RGB: {tuple(color)}")

    # Визуализация уникальных цветов
    plot_unique_colors(unique_colors)

    X = preprocess_colors(unique_colors)
    
    y = np.random.randint(2, size=(X.shape[0],))
    
    X_train = X[:1000]
    y_train = y[:1000]

    nn = NeuralNetwork(input_size=X_train.shape[1])
    
    epoch_loss = []
    
    for epoch in range(1000):
        for i in range(X_train.shape[0]):
            inputs = X_train[i].reshape(1, -1)
            target = y_train[i]
            prediction = nn.neuron.forward(inputs)
            error = target - prediction
            nn.neuron.backward(error)

        if epoch % 100 == 0:
            loss = np.mean((y_train - nn.predict(X_train)) ** 2)
            epoch_loss.append(loss)
            print(f'Epoch {epoch}, Loss: {loss}')
    
    # Визуализация графика потерь
    plot_loss(epoch_loss)
    
    predictions = nn.predict(X_train)
    print('Predictions:', predictions)

if __name__ == '__main__':
    main()
