import numpy as np
import json
import argparse
import os
import sys


class LinearRegression:
    """Упрощенная версия модели для предсказаний."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.scale_y = True

    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        """Стандартизация признаков."""
        if self.X_mean is not None and self.X_std is not None:
            return (X - self.X_mean) / (self.X_std + 1e-8)
        return X

    def load(self, model_path: str):
        """Загружает модель из файла."""
        with open(model_path, 'r') as f:
            model_data = json.load(f)

        self.weights = np.array(model_data['weights'])
        self.bias = model_data['bias']
        self.X_mean = np.array(model_data['X_mean'])
        self.X_std = np.array(model_data['X_std'])
        self.y_mean = model_data['y_mean']
        self.y_std = model_data['y_std']
        self.scale_y = model_data.get('scale_y', True)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает значения в рублях."""
        # Масштабирование признаков
        X_scaled = self._standardize_features(X)

        # Предсказание
        y_pred_scaled = np.dot(X_scaled, self.weights) + self.bias

        # Обратное масштабирование y
        if self.scale_y:
            y_pred = y_pred_scaled * self.y_std + self.y_mean
        else:
            y_pred = y_pred_scaled

        # Преобразование обратно в рубли (из логарифмированных значений)
        y_pred = np.expm1(y_pred)  # exp(y) - 1
        return np.maximum(y_pred, 0)  # Защита от отрицательных значений


def main():
    parser = argparse.ArgumentParser(
        description='Предсказание зарплат с использованием обученной модели'
    )
    parser.add_argument(
        'x_data',
        type=str,
        help='Путь к файлу x_data.npy с признаками'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='resources/linear_regression_model.json',
        help='Путь к файлу с обученной моделью'
    )

    args = parser.parse_args()

    # Проверка существования файлов
    if not os.path.exists(args.x_data):
        print(f"Ошибка: Файл '{args.x_data}' не найден")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"Ошибка: Модель '{args.model_path}' не найдена")
        print("Сначала обучите модель: python train.py")
        sys.exit(1)

    # Загрузка данных
    X = np.load(args.x_data)

    # Загрузка модели
    model = LinearRegression()
    model.load(args.model_path)

    # Предсказание
    predictions = model.predict(X)

    # Вывод предсказаний как список float
    print("[")
    for i, pred in enumerate(predictions):
        if i < len(predictions) - 1:
            print(f"  {pred:.2f},")
        else:
            print(f"  {pred:.2f}")
    print("]")

    # Также возвращаем список для использования в коде
    return predictions.tolist()


if __name__ == "__main__":
    main()