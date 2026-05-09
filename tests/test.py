from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
RESULTS_PATH = BASE_DIR / "models" / "results.csv"

TARGET = "payment_value"


def test_final_dataset_exists():
    assert DATA_PATH.exists(), "Файл data/processed/final_dataset.csv должен существовать"


def test_final_dataset_not_empty():
    df = pd.read_csv(DATA_PATH, nrows=10)
    assert not df.empty, "Итоговый датасет не должен быть пустым"


def test_final_dataset_has_target():
    df = pd.read_csv(DATA_PATH, nrows=5)
    assert TARGET in df.columns, f"В датасете должна быть целевая переменная {TARGET}"


def test_final_dataset_has_features():
    df = pd.read_csv(DATA_PATH, nrows=5)
    feature_columns = [col for col in df.columns if col != TARGET]

    assert len(feature_columns) > 0, "В датасете должны быть признаки кроме целевой переменной"


def test_final_dataset_has_expected_columns():
    df = pd.read_csv(DATA_PATH, nrows=5)

    expected_columns = {
        "payment_value",
        "price",
        "freight_value",
        "payment_installments",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    }

    assert expected_columns.issubset(df.columns), (
        "В датасете должны быть основные признаки для обучения модели"
    )


def test_final_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)

    assert df.isna().sum().sum() == 0, (
        "В финальном датасете не должно быть пропущенных значений"
    )


def test_target_is_not_in_features():
    df = pd.read_csv(DATA_PATH, nrows=5)

    x = df.drop(columns=[TARGET])

    assert TARGET not in x.columns, (
        "Целевая переменная payment_value не должна попадать в признаки"
    )


def test_feature_engineering_columns_exist():
    df = pd.read_csv(DATA_PATH, nrows=5)

    expected_features = {
        "product_volume_cm3",
        "freight_to_price_ratio",
        "price_per_weight",
    }

    assert expected_features.issubset(df.columns), (
        "В датасете должны быть признаки, созданные на этапе feature engineering"
    )


def test_results_file_exists():
    assert RESULTS_PATH.exists(), "Файл models/results.csv должен существовать"


def test_results_file_has_metrics():
    results = pd.read_csv(RESULTS_PATH)

    required_columns = {"model", "RMSE", "MAE", "R2"}

    assert required_columns.issubset(results.columns), (
        "Файл results.csv должен содержать колонки: model, RMSE, MAE, R2"
    )


def test_results_file_has_several_models():
    results = pd.read_csv(RESULTS_PATH)

    assert len(results) >= 5, "В results.csv должно быть минимум 5 моделей"


def test_results_metrics_are_numeric():
    results = pd.read_csv(RESULTS_PATH)

    metric_columns = ["RMSE", "MAE", "R2"]

    for column in metric_columns:
        assert pd.api.types.is_numeric_dtype(results[column]), (
            f"Метрика {column} должна быть числовой"
        )


def test_results_metrics_are_not_empty():
    results = pd.read_csv(RESULTS_PATH)

    metric_columns = ["RMSE", "MAE", "R2"]

    assert not results[metric_columns].isna().any().any(), (
        "В таблице результатов не должно быть пропусков в метриках"
    )


def test_best_model_exists():
    assert MODEL_PATH.exists(), "Файл models/best_model.joblib должен существовать"

def test_best_model_can_predict():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    x = df.drop(columns=[TARGET]).select_dtypes(include=["number"])

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        x = x[expected_features]

    x = x.head(10)

    predictions = model.predict(x)

    assert len(predictions) == 10, "Модель должна возвращать 10 предсказаний"