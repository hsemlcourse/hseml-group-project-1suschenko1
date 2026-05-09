[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — Прогнозирование стоимости заказа

**Студент:** Сущенко Илья Денисович  
**Группа:** БИВ 237

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Предобработка данных](#предобработка-данных)
6. [Моделирование и эксперименты](#моделирование-и-эксперименты)
7. [Результаты](#результаты)
8. [Тестирование](#тестирование)
9. [Отчёт](#отчёт)

## Описание задачи

**Задача:** регрессия.

**Датасет:** Brazilian E-Commerce Public Dataset by Olist (Kaggle).

**Целевая переменная:** `payment_value` — стоимость заказа.

**Метрики качества:**

- RMSE — основная метрика;
- MAE — дополнительная метрика;
- R2 — дополнительная метрика.

Цель проекта — построить модель машинного обучения, которая прогнозирует стоимость заказа на основе данных о заказе, оплате, товаре, продавце и доставке.

Практическая ценность проекта заключается в возможности оценивать ожидаемую стоимость заказа, анализировать продажи и использовать модель как основу для дальнейшей аналитики интернет-магазина.

## Структура репозитория

```
.
├── data
│   ├── processed
│   │   └── final_dataset.csv       # Очищенные и обработанные данные
│   └── raw                         # Исходные файлы Olist
├── models
│   ├── best_model.joblib           # Лучшая обученная модель
│   └── results.csv                 # Таблица результатов экспериментов
├── notebooks
│   ├── 01_eda.ipynb                # Разведочный анализ данных
│   ├── 02_baseline.ipynb           # Baseline-модель
│   └── 03_experiments.ipynb        # Эксперименты с моделями
├── presentation                    # Материалы для презентации
├── report
│   ├── images                      # Изображения для отчёта
│   └── report.md                   # Финальный отчёт
├── src
│   ├── __init__.py
│   ├── preprocessing.py            # Предобработка данных
│   └── train.py                    # Обучение и оценка моделей
├── tests
│   └── test.py                     # Тесты пайплайна
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Запуск

### 1. Клонировать репозиторий

```
git clone https://github.com/hsemlcourse/hseml-group-project-1suschenko1.git
cd hseml-group-project-1suschenko1
```

### 2. Создать виртуальное окружение

Для Windows:

```
python -m venv .venv
.venv\Scripts\activate
```

Для Linux/macOS:

```
python -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```
pip install -r requirements.txt
```

### 4. Запустить предобработку данных

```
python src/preprocessing.py
```

После выполнения создаётся файл:

```
data/processed/final_dataset.csv
```

### 5. Запустить обучение моделей

```
python src/train.py
```

После выполнения создаются файлы:

```
models/best_model.joblib
models/results.csv
```

### 6. Запустить тесты

```
python -m pytest tests
```

Результат запуска тестов:

```
15 tests passed
```

### Проверка качества кода

```
ruff check src tests --line-length 120
```

### 7. Запуск через Docker

```
docker-compose up --build
```

### Проверка качества кода

```
ruff check src tests --line-length 120
```

## Данные

В проекте используется датасет **Brazilian E-Commerce Public Dataset by Olist**.

Исходные данные находятся в папке:

```
data/raw/
```

После предобработки формируется итоговый датасет:

```
data/processed/final_dataset.csv
```

Размер итогового датасета:

```
117250 строк, 28 столбцов
```

Целевая переменная:

```
payment_value
```

Используемые признаки включают исходные числовые признаки и признаки, созданные на этапе feature engineering:

```
customer_zip_code_prefix
order_item_id
price
freight_value
payment_sequential
payment_installments
product_name_lenght
product_description_lenght
product_photos_qty
product_weight_g
product_length_cm
product_height_cm
product_width_cm
seller_zip_code_prefix
order_purchase_month
order_purchase_dayofweek
order_purchase_hour
approval_days
delivery_days
delivery_delay_days
shipping_limit_days
product_volume_cm3
product_density
freight_to_price_ratio
total_item_cost
price_per_weight
```

## Предобработка данных

На этапе предобработки были выполнены следующие действия:

- объединение исходных таблиц Olist;
- обработка дат заказа и доставки;
- удаление дубликатов;
- обработка пропусков;
- удаление выбросов по целевой переменной с помощью 99-го перцентиля;
- выбор числовых признаков;
- формирование итогового датасета `final_dataset.csv`.

Также был выполнен feature engineering. Были созданы новые признаки:

- `order_purchase_month` — месяц оформления заказа;
- `order_purchase_dayofweek` — день недели оформления заказа;
- `order_purchase_hour` — час оформления заказа;
- `approval_days` — время до подтверждения заказа;
- `delivery_days` — фактическое время доставки;
- `delivery_delay_days` — отклонение фактической даты доставки от ожидаемой;
- `shipping_limit_days` — срок до лимита отправки;
- `product_volume_cm3` — объём товара;
- `product_density` — плотность товара;
- `freight_to_price_ratio` — отношение стоимости доставки к цене товара;
- `total_item_cost` — сумма цены товара и доставки;
- `price_per_weight` — цена на единицу веса товара.

Целевая переменная `payment_value` не используется при создании признаков, чтобы избежать утечки данных.

## Моделирование и эксперименты

На этапе CP2 были обучены и сравнены несколько моделей машинного обучения для задачи регрессии.

Были использованы следующие модели:

- Linear Regression;
- Ridge;
- Lasso;
- Random Forest;
- Gradient Boosting;
- Extra Trees;
- Ridge tuned;
- Gradient Boosting tuned;
- Random Forest tuned.

Данные были разделены на обучающую и тестовую выборки в соотношении 80/20:

```
Train: 93800 объектов
Test: 23450 объектов
```

Для всех моделей использовался единый pipeline:

- заполнение пропусков с помощью `SimpleImputer`;
- масштабирование признаков с помощью `StandardScaler`;
- обучение модели;
- оценка на тестовой выборке.

Для улучшения качества был выполнен подбор гиперпараметров для Ridge, Gradient Boosting и Random Forest. Основной метрикой при сравнении моделей была RMSE, так как она сильнее штрафует крупные ошибки прогноза.

## Результаты

| Модель | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Extra Trees | 65.5815 | 28.7206 | 0.8319 |
| Gradient Boosting tuned | 65.6167 | 28.1228 | 0.8318 |
| Random Forest tuned | 65.7032 | 27.4990 | 0.8313 |
| Random Forest | 66.0288 | 28.8796 | 0.8296 |
| Gradient Boosting | 66.5757 | 28.6140 | 0.8268 |
| Linear Regression | 87.4126 | 40.2400 | 0.7014 |
| Ridge tuned | 87.4128 | 40.2396 | 0.7014 |
| Ridge | 87.4144 | 40.2262 | 0.7014 |
| Lasso | 87.4723 | 39.8712 | 0.7010 |

Лучшей моделью стала **Extra Trees**, так как она показала минимальное значение RMSE.

По сравнению с baseline-моделью Linear Regression качество заметно улучшилось:

- RMSE уменьшился с 87.4126 до 65.5815;
- R2 вырос с 0.7014 до 0.8319.

После подбора гиперпараметров качество Gradient Boosting и Random Forest также улучшилось, однако минимальное значение RMSE показала модель Extra Trees. Поэтому именно она выбрана финальной моделью.

Финальная модель сохранена в файл:

```
models/best_model.joblib
```

Результаты экспериментов сохранены в файл:

```
models/results.csv
```

## Тестирование

Для контроля воспроизводимости были добавлены автоматические тесты. Они проверяют:

- наличие итогового датасета;
- наличие целевой переменной;
- наличие основных признаков;
- отсутствие пропусков в финальном датасете;
- наличие признаков feature engineering;
- корректность файла `models/results.csv`;
- наличие метрик RMSE, MAE и R2;
- наличие сохранённой модели;
- возможность выполнения предсказания сохранённой моделью.

Команда запуска тестов:

```\
python -m pytest tests
```

Результат:

```\
15 tests passed
```

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
