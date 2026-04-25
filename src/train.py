import os
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET = "payment_value"


def main():
    # Абсолютный путь к корню проекта
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Путь к датасету
    data_path = os.path.join(base_dir, "data", "processed", "final_dataset.csv")

    print("Загружаю файл:", data_path)

    # Проверка существования файла
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден: {data_path}")

    # Чтение данных
    df = pd.read_csv(data_path)

    print("Размер датасета:", df.shape)

    # Разделение на признаки и таргет
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Берём только числовые признаки
    X = X.select_dtypes(include=["number"])

    # Заполняем пропуски
    X = X.fillna(0)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    # Модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказания
    predictions = model.predict(X_test)

    # Метрики
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n=== Baseline: Linear Regression ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()