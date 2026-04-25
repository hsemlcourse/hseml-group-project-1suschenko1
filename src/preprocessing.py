import os
import pandas as pd


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


def clean_data(df, target="payment_value"):
    # удалить дубликаты
    df = df.drop_duplicates()

    # оставить только числовые признаки
    df = df.select_dtypes(include=["number"])

    # удалить строки без target
    df = df.dropna(subset=[target])

    # заполнить пропуски
    df = df.fillna(0)

    # удалить выбросы (99 перцентиль)
    q99 = df[target].quantile(0.99)
    df = df[df[target] <= q99]

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

    print("Очистка данных...")
    df = clean_data(df)

    print("Сохранение...")
    save_processed_data(df, base_dir)

    print("Готово! Размер датасета:", df.shape)


if __name__ == "__main__":
    main()