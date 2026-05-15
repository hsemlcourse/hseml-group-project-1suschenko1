import os

import pandas as pd


TARGET = "payment_value"


def load_raw_data(base_dir):
    raw_path = os.path.join(base_dir, "data", "raw")

    customers = pd.read_csv(os.path.join(raw_path, "olist_customers_dataset.csv"))
    orders = pd.read_csv(os.path.join(raw_path, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(raw_path, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(raw_path, "olist_order_payments_dataset.csv"))
    products = pd.read_csv(os.path.join(raw_path, "olist_products_dataset.csv"))
    sellers = pd.read_csv(os.path.join(raw_path, "olist_sellers_dataset.csv"))
    translation = pd.read_csv(os.path.join(raw_path, "product_category_name_translation.csv"))

    return customers, orders, items, payments, products, sellers, translation


def merge_data(customers, orders, items, payments, products, sellers, translation):
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(items, on="order_id", how="left")
    df = df.merge(payments, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")
    df = df.merge(translation, on="product_category_name", how="left")

    return df


def add_features(df):
    df = df.copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if "order_purchase_timestamp" in df.columns:
        df["order_purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["order_purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
        df["order_purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    if {"order_approved_at", "order_purchase_timestamp"}.issubset(df.columns):
        df["approval_days"] = (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(df.columns):
        df["delivery_days"] = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    if {"order_estimated_delivery_date", "order_delivered_customer_date"}.issubset(df.columns):
        df["delivery_delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400

    if {"shipping_limit_date", "order_purchase_timestamp"}.issubset(df.columns):
        df["shipping_limit_days"] = (
            df["shipping_limit_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

    if {
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    }.issubset(df.columns):
        df["product_volume_cm3"] = (
            df["product_length_cm"]
            * df["product_height_cm"]
            * df["product_width_cm"]
        )

    if {"product_weight_g", "product_volume_cm3"}.issubset(df.columns):
        df["product_density"] = df["product_weight_g"] / (df["product_volume_cm3"] + 1)

    if {"freight_value", "price"}.issubset(df.columns):
        df["freight_to_price_ratio"] = df["freight_value"] / (df["price"] + 1)
        df["total_item_cost"] = df["price"] + df["freight_value"]

    if {"price", "product_weight_g"}.issubset(df.columns):
        df["price_per_weight"] = df["price"] / (df["product_weight_g"] + 1)

    if {"payment_value", "payment_installments"}.issubset(df.columns):
        df["payment_per_installment"] = (
            df["payment_value"] / (df["payment_installments"] + 1)
        )

    return df


def clean_data(df, target=TARGET):
    df = df.copy()

    df = df.drop_duplicates()

    if target not in df.columns:
        raise ValueError(f"В данных нет целевой переменной: {target}")

    df = df.dropna(subset=[target])

    q99 = df[target].quantile(0.99)
    df = df[df[target] <= q99]

    df = df.select_dtypes(include=["number"])

    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.fillna(0)

    return df


def save_processed_data(df, base_dir):
    processed_path = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    output_file = os.path.join(processed_path, "final_dataset.csv")
    df.to_csv(output_file, index=False)

    print(f"Файл сохранён: {output_file}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Загрузка данных...")
    data = load_raw_data(base_dir)

    print("Объединение таблиц...")
    df = merge_data(*data)

    print("Создание новых признаков...")
    df = add_features(df)

    print("Очистка данных...")
    df = clean_data(df)

    print("Сохранение...")
    save_processed_data(df, base_dir)

    print("Готово! Размер датасета:", df.shape)
    print("Колонки итогового датасета:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
