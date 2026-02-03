#!/usr/bin/env python3
"""
Основной скрипт приложения для предсказания зарплат.
"""
import sys
import os
import subprocess

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python app.py path/to/x_data.npy")
        print("Пример: python app.py x_data.npy")
        sys.exit(1)

    # Получаем путь к файлу с данными
    x_data_path = sys.argv[1]

    # Проверяем существование файла
    if not os.path.exists(x_data_path):
        print(f"Ошибка: Файл '{x_data_path}' не найден")
        sys.exit(1)

    # Запускаем предсказание через predict.py
    try:
        result = subprocess.run(
            ['python', 'predict.py', x_data_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении предсказания: {e}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)