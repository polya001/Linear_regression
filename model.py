"""
Линейная регрессия с мини-батч градиентным спуском.
"""
import numpy as np
from typing import Tuple, Optional
import json
import os


class LinearRegression:
    """Линейная регрессия с обучением на батчах."""
    
    def __init__(self, learning_rate: float = 0.001, n_iterations: int = 1000,
                 batch_size: int = 32, regularization: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.regularization = regularization
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.scale_y = True  # Флаг для масштабирования y

    def _initialize_parameters(self, n_features: int):
        """Инициализирует веса и смещение."""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисляет MSE с L2 регуляризацией."""
        m = len(y_true)
        mse = np.mean((y_true - y_pred) ** 2)
        l2_reg = self.regularization * np.sum(self.weights ** 2)
        return mse + l2_reg

    def _compute_gradients(self, X_batch: np.ndarray, y_batch: np.ndarray,
                          y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """Вычисляет градиенты."""
        m = X_batch.shape[0]

        # Градиенты MSE
        dw = -(2/m) * np.dot(X_batch.T, (y_batch - y_pred))
        db = -(2/m) * np.sum(y_batch - y_pred)

        # Добавляем градиенты L2 регуляризации
        dw += 2 * self.regularization * self.weights

        # Градиентный клиппинг для предотвращения переполнения
        dw = np.clip(dw, -1.0, 1.0)

        return dw, db

    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        """Стандартизация признаков."""
        if not hasattr(self, 'X_mean') or not hasattr(self, 'X_std'):
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            # Защита от деления на ноль
            self.X_std[self.X_std == 0] = 1.0

        return (X - self.X_mean) / (self.X_std + 1e-8)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'LinearRegression':
        """
        Обучает модель на батчах.

        Args:
            X: Матрица признаков обучающей выборки
            y: Целевые значения обучающей выборки (должны быть логарифмированы!)
            X_val: Матрица признаков валидационной выборки
            y_val: Целевые значения валидационной выборки

        Returns:
            Обученная модель
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        # Сохраняем статистики для y (логарифмированные значения)
        self.y_mean = y.mean()
        self.y_std = y.std()

        # Стандартизация признаков
        X_scaled = self._standardize_features(X)

        # Стандартизация y (только если scale_y=True)
        if self.scale_y:
            y_scaled = (y - self.y_mean) / (self.y_std + 1e-8)
        else:
            y_scaled = y.copy()

        if X_val is not None:
            X_val_scaled = self._standardize_features(X_val)
            if self.scale_y:
                y_val_scaled = (y_val - self.y_mean) / (self.y_std + 1e-8)
            else:
                y_val_scaled = y_val.copy()

        # Обучение с мини-батчами
        for iteration in range(self.n_iterations):
            # Перемешивание данных
            indices = np.random.permutation(n_samples)
            X_shuffled = X_scaled[indices]
            y_shuffled = y_scaled[indices]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                # Формирование батча
                end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]

                # Прямое распространение
                y_pred_batch = np.dot(X_batch, self.weights) + self.bias

                # Вычисление потерь
                batch_loss = self._compute_loss(y_batch, y_pred_batch)
                epoch_loss += batch_loss
                n_batches += 1

                # Обратное распространение
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred_batch)

                # Обновление параметров с градиентным клиппингом
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Средние потери за эпоху
            if n_batches > 0:
                epoch_loss /= n_batches
            self.loss_history.append(epoch_loss)

            # Валидация
            if X_val is not None and iteration % 100 == 0:
                y_val_pred = np.dot(X_val_scaled, self.weights) + self.bias
                val_loss = self._compute_loss(y_val_scaled, y_val_pred)

                if iteration == 0 or iteration % 500 == 0:
                    print(f"Iteration {iteration}: Train Loss = {epoch_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}")

        return self

    def predict(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Предсказывает значения.

        Args:
            X: Матрица признаков
            denormalize: Флаг денормализации предсказаний

        Returns:
            Предсказанные значения
        """
        # Масштабирование признаков
        X_scaled = self._standardize_features(X)

        # Предсказание
        y_pred_scaled = np.dot(X_scaled, self.weights) + self.bias

        if denormalize:
            # Обратное масштабирование y
            if self.scale_y:
                y_pred = y_pred_scaled * self.y_std + self.y_mean
            else:
                y_pred = y_pred_scaled

            # Преобразование обратно в рубли (из логарифмированных значений)
            y_pred = np.expm1(y_pred)  # exp(y) - 1, обратное преобразование log1p
            return np.maximum(y_pred, 0)  # Защита от отрицательных значений
        else:
            return y_pred_scaled

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Вычисляет коэффициент детерминации R²."""
        y_pred = self.predict(X)
        y_true_original = np.expm1(y)  # Преобразуем логарифмированные y обратно

        ss_res = np.sum((y_true_original - y_pred) ** 2)
        ss_tot = np.sum((y_true_original - np.mean(y_true_original)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    def save(self, path: str):
        """Сохраняет модель в файл."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'X_mean': self.X_mean.tolist(),
            'X_std': self.X_std.tolist(),
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'batch_size': self.batch_size,
            'regularization': self.regularization,
            'scale_y': self.scale_y
        }

        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LinearRegression':
        """Загружает модель из файла."""
        with open(path, 'r') as f:
            model_data = json.load(f)

        model = cls(
            learning_rate=model_data['learning_rate'],
            n_iterations=model_data['n_iterations'],
            batch_size=model_data['batch_size'],
            regularization=model_data.get('regularization', 0.01)
        )

        model.weights = np.array(model_data['weights'])
        model.bias = model_data['bias']
        model.X_mean = np.array(model_data['X_mean'])
        model.X_std = np.array(model_data['X_std'])
        model.y_mean = model_data['y_mean']
        model.y_std = model_data['y_std']
        model.scale_y = model_data.get('scale_y', True)
        
        return model