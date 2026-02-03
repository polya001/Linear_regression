"""
Вспомогательные функции для обработки данных и оценки модели.
"""
import numpy as np
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_data(X: np.ndarray, y: np.ndarray,
               train_size: float = 0.6, val_size: float = 0.2,
               random_state: int = 42) -> Tuple:
    """
    Разделяет данные на обучающую, валидационную и тестовую выборки.

    Args:
        X: Матрица признаков
        y: Целевая переменная
        train_size: Доля обучающей выборки
        val_size: Доля валидационной выборки
        random_state: Seed для воспроизводимости

    Returns:
        Кортеж: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]

    # Перемешивание индексов
    indices = np.random.permutation(n_samples)

    # Вычисление размеров выборок
    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)

    # Разделение индексов
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Разделение данных
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Вычисляет метрики качества модели.

    Args:
        y_true: Истинные значения (в рублях)
        y_pred: Предсказанные значения (в рублях)

    Returns:
        Словарь с метриками
    """
    # Вычисляем MAPE только для положительных значений
    mask = y_true > 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mape
    }

    return metrics


def print_metrics(metrics: dict, dataset_name: str = ""):
    """
    Выводит метрики в читаемом формате.

    Args:
        metrics: Словарь с метриками
        dataset_name: Название набора данных
    """
    print(f"\n{'=' * 50}")
    if dataset_name:
        print(f"МЕТРИКИ НА {dataset_name.upper()}")
    print(f"{'=' * 50}")

    for metric_name, value in metrics.items():
        if metric_name == 'R2':
            print(f"{metric_name}: {value:.4f}")
        elif metric_name == 'MAPE':
            print(f"{metric_name}: {value:.2f}%")
        else:
            print(f"{metric_name}: {value:,.2f}")


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Подбор гиперпараметров модели.

    Args:
        X_train: Обучающие данные
        y_train: Целевые значения обучающие (логарифмированные)
        X_val: Валидационные данные
        y_val: Целевые значения валидационные (логарифмированные)

    Returns:
        Словарь с лучшими параметрами
    """
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64, 128]
    regularizations = [0.001, 0.01, 0.1]

    best_params = None
    best_score = float('inf')

    print("Подбор гиперпараметров...")
    print("-" * 50)

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for reg in regularizations:
                from model import LinearRegression

                model = LinearRegression(
                    learning_rate=lr,
                    batch_size=batch_size,
                    regularization=reg,
                    n_iterations=500  # Меньше итераций для быстрого подбора
                )

                # Обучаем на обучающей выборке
                model.fit(X_train, y_train)

                # Предсказываем на валидационной выборке
                y_val_pred = model.predict(X_val, denormalize=False)

                # Вычисляем MSE на логарифмированных значениях
                val_loss = np.mean((y_val - y_val_pred) ** 2)

                if val_loss < best_score:
                    best_score = val_loss
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'regularization': reg
                    }

                    print(f"  Новые лучшие параметры: lr={lr}, batch_size={batch_size}, reg={reg}, score={val_loss:.4f}")

    print(f"\nЛучшие параметры: {best_params}")
    print(f"Лучшая ошибка: {best_score:.4f}")

    return best_params