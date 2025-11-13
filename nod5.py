import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Wyłącz ostrzeżenia TensorFlow

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("WARIANT 9 - IMPLEMENTACJA WARSTW SIECI NEURONOWYCH")
print("=" * 80)

print("\n" + "=" * 80)
print("CZĘŚĆ 1: WARSTWA GĘSTA - KLASYFIKACJA IRIS Z DROPOUT 0.5")
print("=" * 80)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

scaler = StandardScaler()
X_train_iris_scaled = scaler.fit_transform(X_train_iris)
X_test_iris_scaled = scaler.transform(X_test_iris)

print(f"\nRozmiar zbioru treningowego: {X_train_iris_scaled.shape}")
print(f"Rozmiar zbioru testowego: {X_test_iris_scaled.shape}")
print(f"Liczba klas: {len(np.unique(y_iris))}")

model_dense = models.Sequential(
    [
        layers.Input(shape=(4,)),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(0.5, name="dropout_1"),  # Dropout 0.5 zgodnie z wariantem
        layers.Dense(32, activation="relu", name="dense_2"),
        layers.Dropout(0.5, name="dropout_2"),  # Drugi Dropout 0.5
        layers.Dense(16, activation="relu", name="dense_3"),
        layers.Dense(3, activation="softmax", name="output"),  # 3 klasy
    ]
)

model_dense.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("\nArchitektura modelu Dense:")
model_dense.summary()

print("\nTrening modelu Dense...")
history_dense = model_dense.fit(
    X_train_iris_scaled,
    y_train_iris,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0,
)

loss_dense, accuracy_dense = model_dense.evaluate(
    X_test_iris_scaled, y_test_iris, verbose=0
)
print(f"\nWyniki na zbiorze testowym:")
print(f"Loss: {loss_dense:.4f}")
print(f"Accuracy: {accuracy_dense:.4f}")

y_pred_iris = model_dense.predict(X_test_iris_scaled, verbose=0)
y_pred_iris_classes = np.argmax(y_pred_iris, axis=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history_dense.history["loss"], label="Loss treningowy")
axes[0, 0].plot(history_dense.history["val_loss"], label="Loss walidacyjny")
axes[0, 0].set_title("Krzywe uczenia - Loss")
axes[0, 0].set_xlabel("Epoka")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history_dense.history["accuracy"], label="Accuracy treningowy")
axes[0, 1].plot(history_dense.history["val_accuracy"], label="Accuracy walidacyjny")
axes[0, 1].set_title("Krzywe uczenia - Accuracy")
axes[0, 1].set_xlabel("Epoka")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)

cm_iris = confusion_matrix(y_test_iris, y_pred_iris_classes)
sns.heatmap(
    cm_iris,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1, 0],
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
axes[1, 0].set_title("Macierz pomyłek - IRIS")
axes[1, 0].set_ylabel("Prawdziwa klasa")
axes[1, 0].set_xlabel("Przewidziana klasa")

axes[1, 1].hist(
    [y_test_iris, y_pred_iris_classes],
    label=["Prawdziwe", "Przewidziane"],
    bins=3,
    alpha=0.7,
)
axes[1, 1].set_title("Rozkład klas")
axes[1, 1].set_xlabel("Klasa")
axes[1, 1].set_ylabel("Liczba próbek")
axes[1, 1].legend()
axes[1, 1].set_xticks([0, 1, 2])
axes[1, 1].set_xticklabels(iris.target_names)

plt.tight_layout()
plt.savefig("wariant9_dense_layer.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nRaport klasyfikacji:")
print(
    classification_report(
        y_test_iris, y_pred_iris_classes, target_names=iris.target_names
    )
)

print("\n" + "=" * 80)
print("CZĘŚĆ 2: WARSTWA KONWOLUCYJNA - CNN DLA MNIST Z WIĘKSZĄ LICZBĄ WARSTW")
print("=" * 80)

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

X_train_mnist = X_train_mnist.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test_mnist = X_test_mnist.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(f"\nRozmiar zbioru treningowego: {X_train_mnist.shape}")
print(f"Rozmiar zbioru testowego: {X_test_mnist.shape}")

model_cnn = models.Sequential(
    [
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(28, 28, 1), name="conv1"
        ),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv3"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv4"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="relu", name="conv5"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu", name="dense1"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", name="dense2"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax", name="output"),
    ]
)

model_cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("\nArchitektura modelu CNN:")
model_cnn.summary()

print("\nTrening modelu CNN...")
history_cnn = model_cnn.fit(
    X_train_mnist,
    y_train_mnist,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_mnist, y_test_mnist, verbose=0)
print(f"\nWyniki na zbiorze testowym:")
print(f"Loss: {loss_cnn:.4f}")
print(f"Accuracy: {accuracy_cnn:.4f}")

y_pred_mnist = model_cnn.predict(X_test_mnist[:1000], verbose=0)
y_pred_mnist_classes = np.argmax(y_pred_mnist, axis=1)

fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 3, 1)
ax1.plot(history_cnn.history["loss"], label="Loss treningowy")
ax1.plot(history_cnn.history["val_loss"], label="Loss walidacyjny")
ax1.set_title("Krzywe uczenia - Loss")
ax1.set_xlabel("Epoka")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(history_cnn.history["accuracy"], label="Accuracy treningowy")
ax2.plot(history_cnn.history["val_accuracy"], label="Accuracy walidacyjny")
ax2.set_title("Krzywe uczenia - Accuracy")
ax2.set_xlabel("Epoka")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True)

ax3 = plt.subplot(2, 3, 3)
cm_mnist = confusion_matrix(y_test_mnist[:1000], y_pred_mnist_classes)
sns.heatmap(cm_mnist, annot=True, fmt="d", cmap="Blues", ax=ax3)
ax3.set_title("Macierz pomyłek - MNIST")
ax3.set_ylabel("Prawdziwa klasa")
ax3.set_xlabel("Przewidziana klasa")

for i in range(3):
    ax = plt.subplot(2, 3, i + 4)
    idx = np.random.randint(0, 1000)
    ax.imshow(X_test_mnist[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Prawda: {y_test_mnist[idx]}, Pred: {y_pred_mnist_classes[idx]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("wariant9_cnn_layer.png", dpi=300, bbox_inches="tight")
plt.show()

layer_outputs = [layer.output for layer in model_cnn.layers if "conv" in layer.name]
activation_model = models.Model(inputs=model_cnn.inputs, outputs=layer_outputs)

sample_image = X_test_mnist[0:1]
activations = activation_model.predict(sample_image, verbose=0)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, activation in enumerate(activations[:6]):
    axes[i].imshow(activation[0, :, :, 0], cmap="viridis")
    axes[i].set_title(f"Warstwa Conv {i+1} - Filtr 1")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("wariant9_cnn_filters.png", dpi=300, bbox_inches="tight")
plt.show()


print("\n" + "=" * 80)
print("CZĘŚĆ 3: WARSTWA REKURENCYJNA - ANALIZA SENTYMENTU IMDB (EMBEDDING=16)")
print("=" * 80)


max_features = 10000
maxlen = 200

print("Wczytywanie danych IMDB...")
(X_train_imdb, y_train_imdb), (X_test_imdb, y_test_imdb) = (
    tf.keras.datasets.imdb.load_data(num_words=max_features)
)

X_train_imdb = pad_sequences(X_train_imdb, maxlen=maxlen)
X_test_imdb = pad_sequences(X_test_imdb, maxlen=maxlen)

print(f"\nRozmiar zbioru treningowego: {X_train_imdb.shape}")
print(f"Rozmiar zbioru testowego: {X_test_imdb.shape}")

inputs = layers.Input(shape=(maxlen,), name="input")
x = layers.Embedding(
    input_dim=max_features, output_dim=16, input_length=maxlen, name="embedding"
)(inputs)
x = layers.LSTM(64, return_sequences=True, name="lstm1")(x)
x = layers.Dropout(0.3, name="dropout1")(x)
x = layers.LSTM(32, name="lstm2")(x)
x = layers.Dropout(0.3, name="dropout2")(x)
x = layers.Dense(32, activation="relu", name="dense1")(x)
x = layers.Dropout(0.5, name="dropout3")(x)
outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

model_rnn = models.Model(inputs=inputs, outputs=outputs, name="LSTM_Model")

model_rnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("\nArchitektura modelu LSTM:")
model_rnn.summary()

print("\nTrening modelu LSTM...")
history_rnn = model_rnn.fit(
    X_train_imdb,
    y_train_imdb,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1,
)

loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_imdb, y_test_imdb, verbose=0)
print(f"\nWyniki na zbiorze testowym:")
print(f"Loss: {loss_rnn:.4f}")
print(f"Accuracy: {accuracy_rnn:.4f}")

y_pred_imdb = model_rnn.predict(X_test_imdb, verbose=0)
y_pred_imdb_classes = (y_pred_imdb > 0.5).astype(int).flatten()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history_rnn.history["loss"], label="Loss treningowy")
axes[0, 0].plot(history_rnn.history["val_loss"], label="Loss walidacyjny")
axes[0, 0].set_title("Krzywe uczenia - Loss (LSTM)")
axes[0, 0].set_xlabel("Epoka")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history_rnn.history["accuracy"], label="Accuracy treningowy")
axes[0, 1].plot(history_rnn.history["val_accuracy"], label="Accuracy walidacyjny")
axes[0, 1].set_title("Krzywe uczenia - Accuracy (LSTM)")
axes[0, 1].set_xlabel("Epoka")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)

cm_imdb = confusion_matrix(y_test_imdb, y_pred_imdb_classes)
sns.heatmap(
    cm_imdb,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1, 0],
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
axes[1, 0].set_title("Macierz pomyłek - IMDB (LSTM)")
axes[1, 0].set_ylabel("Prawdziwa klasa")
axes[1, 0].set_xlabel("Przewidziana klasa")

axes[1, 1].hist(y_pred_imdb, bins=50, edgecolor="black")
axes[1, 1].axvline(x=0.5, color="r", linestyle="--", label="Próg decyzyjny")
axes[1, 1].set_title("Rozkład prawdopodobieństw predykcji (LSTM)")
axes[1, 1].set_xlabel("Prawdopodobieństwo (sentiment pozytywny)")
axes[1, 1].set_ylabel("Liczba próbek")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("wariant9_lstm_layer.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nRaport klasyfikacji LSTM:")
print(
    classification_report(
        y_test_imdb, y_pred_imdb_classes, target_names=["Negative", "Positive"]
    )
)


print("\n" + "=" * 80)
print("CZĘŚĆ 4: WARSTWA TRANSFORMER - TINY TRANSFORMER")
print("=" * 80)


class TinyTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


embed_dim = 32
num_heads = 2
ff_dim = 64

maxlen_transformer = 100

X_train_transformer = pad_sequences(X_train_imdb, maxlen=maxlen_transformer)
X_test_transformer = pad_sequences(X_test_imdb, maxlen=maxlen_transformer)

inputs = layers.Input(shape=(maxlen_transformer,))
embedding_layer = layers.Embedding(max_features, embed_dim)(inputs)
transformer_block = TinyTransformerBlock(embed_dim, num_heads, ff_dim)(embedding_layer)
x = layers.GlobalAveragePooling1D()(transformer_block)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model_transformer = models.Model(inputs=inputs, outputs=outputs)

model_transformer.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

print("\nArchitektura Tiny Transformer:")
model_transformer.summary()

print("\nTrening Tiny Transformer...")
history_transformer = model_transformer.fit(
    X_train_transformer,
    y_train_imdb,
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
)

loss_transformer, accuracy_transformer = model_transformer.evaluate(
    X_test_transformer, y_test_imdb, verbose=0
)
print(f"\nWyniki na zbiorze testowym:")
print(f"Loss: {loss_transformer:.4f}")
print(f"Accuracy: {accuracy_transformer:.4f}")

y_pred_transformer = model_transformer.predict(X_test_transformer, verbose=0)
y_pred_transformer_classes = (y_pred_transformer > 0.5).astype(int).flatten()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history_transformer.history["loss"], label="Loss treningowy")
axes[0, 0].plot(history_transformer.history["val_loss"], label="Loss walidacyjny")
axes[0, 0].set_title("Krzywe uczenia - Loss (Tiny Transformer)")
axes[0, 0].set_xlabel("Epoka")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history_transformer.history["accuracy"], label="Accuracy treningowy")
axes[0, 1].plot(
    history_transformer.history["val_accuracy"], label="Accuracy walidacyjny"
)
axes[0, 1].set_title("Krzywe uczenia - Accuracy (Tiny Transformer)")
axes[0, 1].set_xlabel("Epoka")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)

cm_transformer = confusion_matrix(y_test_imdb, y_pred_transformer_classes)
sns.heatmap(
    cm_transformer,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1, 0],
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
axes[1, 0].set_title("Macierz pomyłek - Tiny Transformer")
axes[1, 0].set_ylabel("Prawdziwa klasa")
axes[1, 0].set_xlabel("Przewidziana klasa")

axes[1, 1].hist(y_pred_transformer, bins=50, edgecolor="black")
axes[1, 1].axvline(x=0.5, color="r", linestyle="--", label="Próg decyzyjny")
axes[1, 1].set_title("Rozkład prawdopodobieństw predykcji")
axes[1, 1].set_xlabel("Prawdopodobieństwo (sentiment pozytywny)")
axes[1, 1].set_ylabel("Liczba próbek")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("wariant9_transformer.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nRaport klasyfikacji Tiny Transformer:")
print(
    classification_report(
        y_test_imdb, y_pred_transformer_classes, target_names=["Negative", "Positive"]
    )
)

print("\n" + "=" * 80)
print("PODSUMOWANIE WYNIKÓW WSZYSTKICH MODELI")
print("=" * 80)

results_summary = {
    "Model": [
        "Dense Layer (IRIS)",
        "CNN (MNIST)",
        "RNN (IMDB)",
        "Tiny Transformer (IMDB)",
    ],
    "Accuracy": [accuracy_dense, accuracy_cnn, accuracy_rnn, accuracy_transformer],
    "Loss": [loss_dense, loss_cnn, loss_rnn, loss_transformer],
}

print("\nTabela porównawcza:")
print("-" * 80)
print(f"{'Model':<30} {'Accuracy':>15} {'Loss':>15}")
print("-" * 80)
for i in range(len(results_summary["Model"])):
    print(
        f"{results_summary['Model'][i]:<30} {results_summary['Accuracy'][i]:>15.4f} {results_summary['Loss'][i]:>15.4f}"
    )
print("-" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.bar(
    results_summary["Model"],
    results_summary["Accuracy"],
    color=["blue", "green", "orange", "red"],
)
ax1.set_title("Porównanie Accuracy")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.tick_params(axis="x", rotation=45)
ax1.grid(True, alpha=0.3)

ax2.bar(
    results_summary["Model"],
    results_summary["Loss"],
    color=["blue", "green", "orange", "red"],
)
ax2.set_title("Porównanie Loss")
ax2.set_ylabel("Loss")
ax2.tick_params(axis="x", rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("wariant9_summary.png", dpi=300, bbox_inches="tight")
plt.show()
