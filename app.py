import os

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")

FEATURES = [
    "customer_zip_code_prefix",
    "order_item_id",
    "price",
    "freight_value",
    "payment_sequential",
    "payment_installments",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "seller_zip_code_prefix",
    "order_purchase_month",
    "order_purchase_dayofweek",
    "order_purchase_hour",
    "approval_days",
    "delivery_days",
    "delivery_delay_days",
    "shipping_limit_days",
    "product_volume_cm3",
    "product_density",
    "freight_to_price_ratio",
    "total_item_cost",
    "price_per_weight",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Прогноз стоимости заказа",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Прогнозирование стоимости заказа")
st.write(
    "Приложение использует обученную ML-модель для прогноза "
    "целевой переменной `payment_value`."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Файл модели не найден. Сначала запусти обучение: "
        "`python src/train.py`"
    )
    st.stop()

model = load_model()

st.sidebar.header("Параметры заказа")

price = st.sidebar.number_input("Цена товара", min_value=0.0, value=100.0)
freight_value = st.sidebar.number_input("Стоимость доставки", min_value=0.0, value=20.0)
payment_installments = st.sidebar.number_input("Количество платежей", min_value=1, value=1)
product_weight_g = st.sidebar.number_input("Вес товара, г", min_value=0.0, value=500.0)
product_length_cm = st.sidebar.number_input("Длина товара, см", min_value=0.0, value=20.0)
product_height_cm = st.sidebar.number_input("Высота товара, см", min_value=0.0, value=10.0)
product_width_cm = st.sidebar.number_input("Ширина товара, см", min_value=0.0, value=15.0)

order_purchase_month = st.sidebar.slider("Месяц заказа", 1, 12, 5)
order_purchase_dayofweek = st.sidebar.slider("День недели заказа", 0, 6, 2)
order_purchase_hour = st.sidebar.slider("Час заказа", 0, 23, 14)

delivery_days = st.sidebar.number_input("Срок доставки, дней", value=10.0)
delivery_delay_days = st.sidebar.number_input("Задержка доставки, дней", value=0.0)
approval_days = st.sidebar.number_input("Время подтверждения заказа, дней", value=1.0)
shipping_limit_days = st.sidebar.number_input("Срок до лимита отправки, дней", value=3.0)

product_volume_cm3 = product_length_cm * product_height_cm * product_width_cm
product_density = product_weight_g / (product_volume_cm3 + 1)
freight_to_price_ratio = freight_value / (price + 1)
total_item_cost = price + freight_value
price_per_weight = price / (product_weight_g + 1)

input_data = {
    "customer_zip_code_prefix": 0,
    "order_item_id": 1,
    "price": price,
    "freight_value": freight_value,
    "payment_sequential": 1,
    "payment_installments": payment_installments,
    "product_name_lenght": 40,
    "product_description_lenght": 500,
    "product_photos_qty": 1,
    "product_weight_g": product_weight_g,
    "product_length_cm": product_length_cm,
    "product_height_cm": product_height_cm,
    "product_width_cm": product_width_cm,
    "seller_zip_code_prefix": 0,
    "order_purchase_month": order_purchase_month,
    "order_purchase_dayofweek": order_purchase_dayofweek,
    "order_purchase_hour": order_purchase_hour,
    "approval_days": approval_days,
    "delivery_days": delivery_days,
    "delivery_delay_days": delivery_delay_days,
    "shipping_limit_days": shipping_limit_days,
    "product_volume_cm3": product_volume_cm3,
    "product_density": product_density,
    "freight_to_price_ratio": freight_to_price_ratio,
    "total_item_cost": total_item_cost,
    "price_per_weight": price_per_weight,
}

df_input = pd.DataFrame([input_data])

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    df_input = df_input[model_features]
else:
    df_input = df_input[FEATURES]

st.subheader("Входные данные")
st.dataframe(df_input)

if st.button("Предсказать стоимость заказа"):
    prediction = model.predict(df_input)[0]

    st.success(f"Прогнозируемая стоимость заказа: {prediction:.2f}")