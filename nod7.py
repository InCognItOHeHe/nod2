import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def generate_soil_moisture(n_samples=1000):
    x = np.arange(n_samples)
    y = 0.5 + 0.3 * np.sin(0.02 * x) + np.random.normal(0, 0.05, n_samples)
    return y.reshape(-1, 1)


def create_dataset(dataset, look_back=20):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


print("=== (a) Prognozowanie wilgotności gleby ===")

data = generate_soil_moisture()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

look_back = 20
X, y = create_dataset(data_scaled, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)

model_a = Sequential([Input(shape=(look_back, 1)), LSTM(50), Dense(1)])

model_a.compile(optimizer="adam", loss="mse")
model_a.fit(X, y, epochs=20, batch_size=16, verbose=1)

pred = model_a.predict(X)
pred = scaler.inverse_transform(pred)
real = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure()
plt.plot(real, label="Rzeczywista wilgotność")
plt.plot(pred, label="Prognozowana wilgotność")
plt.legend()
plt.title("Prognozowanie wilgotności gleby")
plt.show()


def generate_drone_data(n_samples=1000, timesteps=15):
    X = np.random.normal(0, 1, (n_samples, timesteps))
    y = np.zeros(n_samples)

    anomaly_idx = np.random.choice(n_samples, n_samples // 10, replace=False)
    X[anomaly_idx] += np.random.normal(4, 1, (len(anomaly_idx), timesteps))
    y[anomaly_idx] = 1

    return X.reshape(n_samples, timesteps, 1), y


print("\n=== (b) Wykrywanie anomalii w danych z dronów ===")

X, y = generate_drone_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_b = Sequential(
    [Input(shape=(X.shape[1], 1)), LSTM(32), Dense(1, activation="sigmoid")]
)

model_b.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model_b.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
)

loss, acc = model_b.evaluate(X_test, y_test)
print(f"Dokładność wykrywania anomalii: {acc:.2f}")


def generate_plant_growth_data(samples_per_class=500, timesteps=20):
    X, y = [], []

    for cls in range(3):
        for _ in range(samples_per_class):
            if cls == 0:  # kiełkowanie
                seq = np.random.normal(0.2, 0.05, timesteps)
            elif cls == 1:  # wzrost
                seq = np.linspace(0.2, 0.8, timesteps) + np.random.normal(
                    0, 0.05, timesteps
                )
            else:  # kwitnienie
                seq = np.sin(np.linspace(0, 3 * np.pi, timesteps)) + 1

            X.append(seq)
            y.append(cls)

    X = np.array(X).reshape(-1, timesteps, 1)
    y = tf.keras.utils.to_categorical(y, num_classes=3)
    return X, y


print("\n=== (c) Rozpoznawanie faz wzrostu roślin ===")

X, y = generate_plant_growth_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_c = Sequential(
    [Input(shape=(X.shape[1], 1)), LSTM(64), Dense(3, activation="softmax")]
)

model_c.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_c.fit(
    X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test)
)

loss, acc = model_c.evaluate(X_test, y_test)
print(f"Dokładność rozpoznawania faz wzrostu: {acc:.2f}")
