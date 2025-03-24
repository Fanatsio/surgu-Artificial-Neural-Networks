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

split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]
train_target, test_target = target[:split_idx], target[split_idx:]

model = Sequential()
model.add(Dense(units=16, input_dim=2, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='tanh'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'mae', 'mse'])

history = model.fit(train_data, train_target, epochs=300, batch_size=32, validation_split=0.2, verbose=1)

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

train_loss, train_accuracy, train_mae, train_mse = model.evaluate(train_data, train_target, verbose=0)
test_loss, test_accuracy, test_mae, test_mse = model.evaluate(test_data, test_target, verbose=0)
full_accuracy = np.mean(predictions_rounded.flatten() == target) * 100

print("\nФинальные метрики:")
print(f"Тренировочная выборка:")
print(f"Loss: {train_loss:.4f}")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"\nТестовая выборка:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"\nТочность на всей окружности: {full_accuracy:.2f}%")