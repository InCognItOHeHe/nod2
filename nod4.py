import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# CZĘŚĆ 1: DEFINICJA FUNKCJI CELU


def f_original(x, y):
    """Funkcja oryginalna: f(x,y) = (x² - y²) / (1 + x² + y²)"""
    return (x**2 - y**2) / (1 + x**2 + y**2)


def f_alternative(x, y):
    """Funkcja alternatywna (nieliniowa): f(x,y) = sin(x) * cos(y) + 0.5 * (x² + y²)"""
    return np.sin(x) * np.cos(y) + 0.5 * (x**2 + y**2)


def grad_f_original(x, y):
    """Oblicza gradient funkcji oryginalnej"""
    denominator = (1 + x**2 + y**2) ** 2
    df_dx = (2 * x + 4 * x * y**2) / denominator
    df_dy = (-2 * y - 4 * x**2 * y) / denominator
    return np.array([df_dx, df_dy])


def grad_f_alternative(x, y):
    """Oblicza gradient funkcji alternatywnej"""
    df_dx = np.cos(x) * np.cos(y) + x
    df_dy = -np.sin(x) * np.sin(y) + y
    return np.array([df_dx, df_dy])


# CZĘŚĆ 2: ALGORYTMY OPTYMALIZACJI


def gradient_descent(f, grad_f, x0, eta, max_iter=1000, tol=1e-6):
    """Gradient Descent"""
    x = x0.copy()
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x[0], x[1])
        x_new = x - eta * grad
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return np.array(history)


def momentum(f, grad_f, x0, eta, beta=0.9, max_iter=1000, tol=1e-6):
    """Gradient Descent z Momentum"""
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x[0], x[1])
        v = beta * v + eta * grad
        x_new = x - v
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return np.array(history)


def adam(
    f, grad_f, x0, eta, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000, tol=1e-6
):
    """Adam Optimizer"""
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [x.copy()]

    for t in range(1, max_iter + 1):
        grad = grad_f(x[0], x[1])

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x_new = x - eta * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return np.array(history)


# CZĘŚĆ 3: WIZUALIZACJA


def plot_optimization_paths(f, histories, etas, title, func_name):
    """Wizualizuje ścieżki optymalizacji dla różnych algorytmów"""
    fig = plt.figure(figsize=(18, 5))

    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    algorithms = ["Gradient Descent", "Momentum", "Adam"]

    for idx, eta in enumerate(etas):
        ax = fig.add_subplot(1, 3, idx + 1)

        contour = ax.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)

        colors = ["red", "blue", "green"]
        for i, alg_name in enumerate(algorithms):
            history = histories[alg_name][idx]
            ax.plot(
                history[:, 0],
                history[:, 1],
                "o-",
                color=colors[i],
                label=alg_name,
                markersize=3,
                linewidth=1.5,
            )
            ax.plot(
                history[0, 0],
                history[0, 1],
                "ko",
                markersize=8,
                label="Start" if i == 0 else "",
            )
            ax.plot(
                history[-1, 0],
                history[-1, 1],
                "k*",
                markersize=12,
                label="Koniec" if i == 0 else "",
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"η = {eta}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{title}\nFunkcja: {func_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_convergence(histories, etas, f, title):
    """Wykres zbieżności - wartość funkcji celu w czasie"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    algorithms = ["Gradient Descent", "Momentum", "Adam"]
    colors = ["red", "blue", "green"]

    for idx, eta in enumerate(etas):
        ax = axes[idx]

        for i, alg_name in enumerate(algorithms):
            history = histories[alg_name][idx]
            values = [f(point[0], point[1]) for point in history]
            ax.plot(values, color=colors[i], label=alg_name, linewidth=2)

        ax.set_xlabel("Iteracja")
        ax.set_ylabel("Wartość funkcji celu")
        ax.set_title(f"η = {eta}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("symlog")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_3d_surface(f, title):
    """Wykres 3D funkcji"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# CZĘŚĆ 4: EKSPERYMENTY Z OPTYMALIZACJĄ


def run_experiments(f, grad_f, func_name):
    """Przeprowadza eksperymenty dla danej funkcji"""
    print(f"\n{'='*70}")
    print(f"EKSPERYMENTY DLA: {func_name}")
    print(f"{'='*70}\n")

    etas = [0.1, 0.05, 0.01]
    x0 = np.array([2.0, 2.0])

    histories = {"Gradient Descent": [], "Momentum": [], "Adam": []}

    plot_3d_surface(f, f"Funkcja: {func_name}")

    for eta in etas:
        print(f"\nTestowanie η = {eta}:")

        hist_gd = gradient_descent(f, grad_f, x0, eta)
        histories["Gradient Descent"].append(hist_gd)
        final_val_gd = f(hist_gd[-1, 0], hist_gd[-1, 1])
        print(
            f"  Gradient Descent: {len(hist_gd)} iteracji, wartość końcowa: {final_val_gd:.6f}"
        )

        hist_mom = momentum(f, grad_f, x0, eta)
        histories["Momentum"].append(hist_mom)
        final_val_mom = f(hist_mom[-1, 0], hist_mom[-1, 1])
        print(
            f"  Momentum: {len(hist_mom)} iteracji, wartość końcowa: {final_val_mom:.6f}"
        )

        hist_adam = adam(f, grad_f, x0, eta)
        histories["Adam"].append(hist_adam)
        final_val_adam = f(hist_adam[-1, 0], hist_adam[-1, 1])
        print(
            f"  Adam: {len(hist_adam)} iteracji, wartość końcowa: {final_val_adam:.6f}"
        )

    plot_optimization_paths(
        f,
        histories,
        etas,
        "Ścieżki optymalizacji dla różnych współczynników uczenia",
        func_name,
    )
    plot_convergence(histories, etas, f, f"Zbieżność algorytmów - {func_name}")

    return histories


# CZĘŚĆ 5: SIEĆ NEURONOWA MLP (PyTorch)


class MLP(nn.Module):
    """Model MLP w PyTorch"""

    def __init__(self, input_dim=2):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_pytorch_mlp(model, train_loader, val_loader, eta, epochs=100):
    """Trenuje model PyTorch"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=eta)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Trenowanie
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

    return train_losses, val_losses


def evaluate_pytorch_mlp(model, test_loader):
    """Ocenia model PyTorch"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            mae = torch.mean(torch.abs(outputs.squeeze() - y_batch))
            total_loss += loss.item()
            total_mae += mae.item()

    return total_loss / len(test_loader), total_mae / len(test_loader)


def train_and_evaluate_mlp(f, func_name, etas=[0.1, 0.05, 0.01]):
    """Trenuje i ocenia MLP z różnymi współczynnikami uczenia"""
    print(f"\n{'='*70}")
    print(f"TRENOWANIE MLP DLA: {func_name}")
    print(f"{'='*70}\n")

    np.random.seed(42)
    n_samples = 5000
    x = np.random.uniform(-3, 3, n_samples)
    y = np.random.uniform(-3, 3, n_samples)
    z = f(x, y)

    X = np.column_stack([x, y])
    X_train, X_test, y_train, y_test = train_test_split(
        X, z, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, eta in enumerate(etas):
        print(f"\nTrenowanie z η = {eta}:")

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_val), torch.FloatTensor(y_train_val)
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = MLP()
        train_losses, val_losses = train_pytorch_mlp(
            model, train_loader, val_loader, eta
        )

        test_loss, test_mae = evaluate_pytorch_mlp(model, test_loader)
        print(f"  Test Loss (MSE): {test_loss:.6f}")
        print(f"  Test MAE: {test_mae:.6f}")

        results[eta] = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_loss": test_loss,
            "test_mae": test_mae,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
        }

        ax = axes[idx]
        ax.plot(train_losses, label="Train Loss", linewidth=2)
        ax.plot(val_losses, label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"η = {eta}\nTest MAE: {test_mae:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.suptitle(
        f"Historia trenowania MLP - {func_name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    best_eta = min(results.keys(), key=lambda k: results[k]["test_mae"])
    best_result = results[best_eta]
    best_model = best_result["model"]
    scaler_X = best_result["scaler_X"]
    scaler_y = best_result["scaler_y"]

    print(f"\nNajlepszy model: η = {best_eta}")

    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    grid_points_scaled = scaler_X.transform(grid_points)

    best_model.eval()
    with torch.no_grad():
        predictions_scaled = best_model(torch.FloatTensor(grid_points_scaled)).numpy()

    predictions = scaler_y.inverse_transform(predictions_scaled)
    Z_pred = predictions.reshape(X_grid.shape)
    Z_true = f(X_grid, Y_grid)

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(X_grid, Y_grid, Z_true, cmap="viridis", alpha=0.8)
    ax1.set_title("Prawdziwa funkcja")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")

    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(X_grid, Y_grid, Z_pred, cmap="plasma", alpha=0.8)
    ax2.set_title(f"Predykcja MLP (η = {best_eta})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f(x,y)")

    ax3 = fig.add_subplot(133, projection="3d")
    error = np.abs(Z_true - Z_pred)
    surf3 = ax3.plot_surface(X_grid, Y_grid, error, cmap="Reds", alpha=0.8)
    ax3.set_title("Błąd bezwzględny")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("|error|")

    plt.suptitle(
        f"Porównanie MLP z prawdziwą funkcją - {func_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WARIANT 9 - OPTYMALIZACJA I SIECI NEURONOWE")
    print("=" * 70)

    print("\n\n### CZĘŚĆ 1: FUNKCJA ORYGINALNA ###")
    histories_original = run_experiments(
        f_original, grad_f_original, "f(x,y) = (x² - y²) / (1 + x² + y²)"
    )

    print("\n\n### CZĘŚĆ 2: FUNKCJA ALTERNATYWNA (NIELINIOWA) ###")
    histories_alternative = run_experiments(
        f_alternative, grad_f_alternative, "f(x,y) = sin(x)·cos(y) + 0.5(x² + y²)"
    )

    print("\n\n### CZĘŚĆ 3: SIECI NEURONOWE MLP ###")

    results_mlp_original = train_and_evaluate_mlp(
        f_original, "f(x,y) = (x² - y²) / (1 + x² + y²)", etas=[0.1, 0.05, 0.01]
    )

    results_mlp_alternative = train_and_evaluate_mlp(
        f_alternative, "f(x,y) = sin(x)·cos(y) + 0.5(x² + y²)", etas=[0.1, 0.05, 0.01]
    )

    print("\n" + "=" * 70)
    print("EKSPERYMENTY ZAKOŃCZONE!")
    print("=" * 70)
