import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

theta = np.linspace(-np.pi, np.pi, 200)
radius = 5.0
x_data = radius * np.cos(theta)
y_data = radius * np.sin(theta)

target = np.where(y_data > 0, 1, np.where(y_data < 0, -1, 0))

data = np.column_stack((x_data, y_data))

split_index = int(0.8 * len(data))
train_data, test_data = data[:split_index], data[split_index:]
train_target, test_target = target[:split_index], target[split_index:]

model = Sequential()
model.add(Dense(units=16, input_dim=2, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='tanh'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mae', 'mse'])

_ = model.fit(train_data, train_target, epochs=300, batch_size=32, validation_split=0.2, verbose=1)

predictions = model.predict(data)
predictions_rounded = np.where(predictions > 0.5, 1, np.where(predictions < -0.5, -1, 0))

plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], c=target, cmap='coolwarm', label='Истинные значения', alpha=0.5)
plt.scatter(data[:, 0], data[:, 1], c=predictions_rounded, cmap='coolwarm', marker='x', label='Предсказания', alpha=0.8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Классификация точек полной окружности')
plt.legend()
plt.axis('equal')
plt.colorbar(label='Значение')
plt.grid(True)
plt.show()

full_accuracy = np.mean(predictions_rounded.flatten() == target) * 100

print(f"\nТочность на всей окружности: {full_accuracy:.2f}%")