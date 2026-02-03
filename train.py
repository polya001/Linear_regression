"""
Скрипт для обучения линейной регрессии на данных hh.ru.
"""
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LinearRegression
from utils import split_data, evaluate_model, print_metrics, hyperparameter_tuning


def load_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Загружает данные из .npy файлов."""
    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Загружено данных:")
    print(f"  X: {X.shape[0]:,} образцов, {X.shape[1]} признаков")
    print(f"  y: {y.shape[0]:,} значений")

    return X, y


def main():
    parser = argparse.ArgumentParser(
        description='Обучение линейной регрессии на данных hh.ru'
    )
    parser.add_argument(
        '--x-data',
        type=str,
        default='x_data.npy',
        help='Путь к файлу с признаками (по умолчанию: x_data.npy)'
    )
    parser.add_argument(
        '--y-data',
        type=str,
        default='y_data.npy',
        help='Путь к файлу с целевой переменной (по умолчанию: y_data.npy)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='resources',
        help='Директория для сохранения модели (по умолчанию: resources)'
    )
    parser.add_argument(
        '--no-hyperparam-tuning',
        action='store_true',
        help='Отключить подбор гиперпараметров'
    )

    args = parser.parse_args()

    # Создаем директорию для сохранения модели
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ОБУЧЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ")
    print("=" * 60)

    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    X, y = load_data(args.x_data, args.y_data)

    # Проверяем, что y содержит логарифмированные значения
    print(f"  Минимальное значение y: {y.min():.2f}")
    print(f"  Максимальное значение y: {y.max():.2f}")
    print(f"  Среднее значение y: {y.mean():.2f}")

    # 2. Разделение данных
    print("\n2. Разделение данных (60/20/20)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_size=0.6, val_size=0.2
    )

    print(f"  Обучающая выборка: {X_train.shape[0]:,} образцов")
    print(f"  Валидационная выборка: {X_val.shape[0]:,} образцов")
    print(f"  Тестовая выборка: {X_test.shape[0]:,} образцов")

    # 3. Подбор гиперпараметров (опционально)
    if not args.no_hyperparam_tuning:
        print("\n3. Подбор гиперпараметров...")
        best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val)

        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']
        regularization = best_params['regularization']
    else:
        # Используем значения по умолчанию
        learning_rate = 0.001
        batch_size = 32
        regularization = 0.01

    # 4. Обучение финальной модели
    print("\n4. Обучение финальной модели...")
    print(f"  Параметры: lr={learning_rate}, batch_size={batch_size}, reg={regularization}")

    # Объединяем обучающую и валидационную выборки для финального обучения
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    model = LinearRegression(
        learning_rate=learning_rate,
        batch_size=batch_size,
        regularization=regularization,
        n_iterations=2000
    )

    model.fit(X_train_full, y_train_full, X_val, y_val)

    # 5. Оценка на тестовой выборке
    print("\n5. Оценка модели...")

    # Предсказания (денормализованные - в рублях)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Преобразуем истинные значения обратно в рубли
    y_train_original = np.expm1(y_train)
    y_val_original = np.expm1(y_val)
    y_test_original = np.expm1(y_test)

    # Вычисление метрик на оригинальных значениях (в рублях)
    train_metrics = evaluate_model(y_train_original, y_train_pred)
    val_metrics = evaluate_model(y_val_original, y_val_pred)
    test_metrics = evaluate_model(y_test_original, y_test_pred)
    
    print_metrics(train_metrics, "обучающей выборке")
    print_metrics(val_metrics, "валидационной выборке")
    print_metrics(test_metrics, "тестовой выборке")
    
    # 6. Сохранение модели
    print("\n6. Сохранение модели...")
    model_path = os.path.join(args.output_dir, 'linear_regression_model.json')
    model.save(model_path)
    
    print(f"  Модель сохранена: {model_path}")
    print(f"  Размер файла: {os.path.getsize(model_path) / 1024:.1f} КБ")
    
    # 7. Сохранение метрик
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("МЕТРИКИ ЛИНЕЙНОЙ РЕГРЕССИИ\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Обучающая выборка:\n")
        for metric, value in train_metrics.items():
            f.write(f"  {metric}: {value}\n")
        
        f.write("\nВалидационная выборка:\n")
        for metric, value in val_metrics.items():
            f.write(f"  {metric}: {value}\n")
        
        f.write("\nТестовая выборка:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value}\n")
    
    print(f"  Метрики сохранены: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()