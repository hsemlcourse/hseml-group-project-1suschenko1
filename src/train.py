import os

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET = "payment_value"


def evaluate_model(name, model, x_test, y_test):
    predictions = model.predict(x_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def make_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(base_dir, "data", "processed", "final_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "best_model.joblib")
    results_path = os.path.join(models_dir, "results.csv")

    print("Загружаю файл:", data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Файл не найден: {data_path}\n"
            f"Сначала запусти: python src/preprocessing.py"
        )

    df = pd.read_csv(data_path)

    print("Размер датасета:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(f"В датасете нет целевой переменной: {TARGET}")

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x = x.select_dtypes(include=["number"])

    print("X shape:", x.shape)
    print("y shape:", y.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Train:", x_train.shape)
    print("Test:", x_test.shape)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nОбучаю модель: {name}")

        pipeline = make_pipeline(model)
        pipeline.fit(x_train, y_train)

        metrics = evaluate_model(name, pipeline, x_test, y_test)
        results.append(metrics)
        fitted_models[name] = pipeline

    print("\nПодбор гиперпараметров для Ridge...")

    ridge_pipeline = make_pipeline(Ridge(random_state=RANDOM_STATE))

    ridge_param_grid = {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
    }

    ridge_grid_search = GridSearchCV(
        ridge_pipeline,
        ridge_param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )

    ridge_grid_search.fit(x_train, y_train)

    ridge_tuned_metrics = evaluate_model(
        "Ridge tuned",
        ridge_grid_search.best_estimator_,
        x_test,
        y_test,
    )

    results.append(ridge_tuned_metrics)
    fitted_models["Ridge tuned"] = ridge_grid_search.best_estimator_

    print("Best Ridge params:", ridge_grid_search.best_params_)

    print("\nПодбор гиперпараметров для Gradient Boosting...")

    gb_pipeline = make_pipeline(
        GradientBoostingRegressor(random_state=RANDOM_STATE)
    )

    gb_param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__min_samples_split": [2, 5],
    }

    gb_grid_search = GridSearchCV(
        gb_pipeline,
        gb_param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )

    gb_grid_search.fit(x_train, y_train)

    gb_tuned_metrics = evaluate_model(
        "Gradient Boosting tuned",
        gb_grid_search.best_estimator_,
        x_test,
        y_test,
    )

    results.append(gb_tuned_metrics)
    fitted_models["Gradient Boosting tuned"] = gb_grid_search.best_estimator_

    print("Best Gradient Boosting params:", gb_grid_search.best_params_)

    print("\nПодбор гиперпараметров для Random Forest...")

    rf_pipeline = make_pipeline(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    )

    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    rf_random_search = RandomizedSearchCV(
        rf_pipeline,
        rf_param_grid,
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_random_search.fit(x_train, y_train)

    rf_tuned_metrics = evaluate_model(
        "Random Forest tuned",
        rf_random_search.best_estimator_,
        x_test,
        y_test,
    )

    results.append(rf_tuned_metrics)
    fitted_models["Random Forest tuned"] = rf_random_search.best_estimator_

    print("Best Random Forest params:", rf_random_search.best_params_)

    results_df = pd.DataFrame(results).sort_values("RMSE")

    print("\n=== Results ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model, model_path)
    results_df.to_csv(results_path, index=False)

    print("\n=== Best model ===")
    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()