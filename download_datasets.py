import os
import pandas as pd
import numpy as np
import ssl  # <--- Добавлено

# --- FIX FOR MACOS SSL ERROR ---
# Отключаем проверку сертификатов для скачивания данных
ssl._create_default_https_context = ssl._create_unverified_context
# -------------------------------

from sklearn.datasets import fetch_openml, fetch_california_housing

# Конфигурация
OUTPUT_DIR = "datasets"
TARGET_ROWS = 10000
RANDOM_STATE = 42

# Создаем папку для данных
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_and_save(name, df, target_rows=TARGET_ROWS):
    """
    Очищает датасет, семплирует строки и сохраняет в CSV.
    """
    # 1. Удаление пропусков
    initial_rows = len(df)
    df = df.dropna()

    # 2. Семплирование (если строк больше, чем нужно)
    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=RANDOM_STATE).reset_index(
            drop=True)
    else:
        print(
            f"⚠️ {name}: строк меньше целевого значения ({len(df)} < {target_rows})")

    # 3. Сохранение
    file_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(file_path, index=False)

    # Отчет
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    num_cols = df.select_dtypes(include=['number']).columns

    print(f"✅ {name:<20} | Saved: {len(df)} rows | Cols: {df.shape[1]} "
          f"(Num: {len(num_cols)}, Cat: {len(cat_cols)})")


# --- ГРУППА 1: ЧИСЛОВЫЕ (NUMERICAL) ---

print("\nDownloading Numerical Datasets...")

# 1. Magic Gamma Telescope
# OpenML ID: 1120
data = fetch_openml(data_id=1120, as_frame=True, parser='auto')
df = pd.concat([data.data, data.target], axis=1)
process_and_save("magic_gamma", df)

# 2. California Housing (Sklearn built-in)
california = fetch_california_housing(as_frame=True)
df = pd.concat([california.data, california.target], axis=1)
process_and_save("california_housing", df)

# 3. Letter Recognition
# OpenML ID: 6
data = fetch_openml(data_id=6, as_frame=True, parser='auto')
df = pd.concat([data.data, data.target], axis=1)
process_and_save("letter_recognition", df)

# --- ГРУППА 2: СМЕШАННЫЕ (MIXED) ---

print("\nDownloading Mixed Datasets...")

# 4. Adult (Census Income)
# OpenML ID: 1590
data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
df = pd.concat([data.data, data.target], axis=1)
# Удаляем технические колонки, если есть (fnlwgt часто удаляют, но оставим для чистоты)
process_and_save("adult", df)

# 5. Bank Marketing
# OpenML ID: 1461
data = fetch_openml(data_id=1461, as_frame=True, parser='auto')
df = pd.concat([data.data, data.target], axis=1)
process_and_save("bank_marketing", df)

# 6. Default of Credit Card Clients
# OpenML ID: 42477
data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
df = pd.concat([data.data, data.target], axis=1)
process_and_save("default_credit", df)

# 7. Online Shoppers Purchasing Intention
# Прямая загрузка с UCI, так как на OpenML версии могут отличаться
url_shoppers = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
try:
    df = pd.read_csv(url_shoppers)
    process_and_save("online_shoppers", df)
except Exception as e:
    print(f"❌ Failed to load Online Shoppers: {e}")

# --- ГРУППА 3: КАТЕГОРИАЛЬНЫЕ (CATEGORICAL) ---

print("\nDownloading Categorical Datasets...")

# 8. Nursery
# OpenML ID: 26
# Rows: ~12960, Cols: 8 (All Categorical)
data = fetch_openml(data_id=26, as_frame=True, parser='auto')
df = pd.concat([data.data, data.target], axis=1)
process_and_save("nursery", df)

# 9. Connect-4 (Замена для Chess)
# OpenML ID: 40668
# Rows: ~67557, Cols: 42 (All Categorical)
# Это отличный датасет: полностью категориальный и очень большой
try:
    data = fetch_openml(data_id=40668, as_frame=True, parser='auto')
    df = pd.concat([data.data, data.target], axis=1)
    process_and_save("connect_4", df)
except Exception as e:
    print(f"❌ Failed to load Connect-4: {e}")

# 10. Phishing Websites
# OpenML ID: 4534
# Rows: ~11055, Cols: 30 (All Categorical features encoded as -1, 0, 1)
try:
    data = fetch_openml(data_id=4534, as_frame=True, parser='auto')
    df = pd.concat([data.data, data.target], axis=1)
    process_and_save("phishing_websites", df)
except Exception as e:
    print(
        f"❌ Failed to load Phishing Websites via OpenML. Trying fallback URL...")
    # Прямая ссылка на UCI версию, если OpenML недоступен
    url_phish = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    from scipy.io import arff
    import urllib.request
    import io

    # Скачиваем и читаем ARFF
    resp = urllib.request.urlopen(url_phish)
    data, meta = arff.loadarff(io.StringIO(resp.read().decode('utf-8')))
    df = pd.DataFrame(data)
    # Декодируем байтовые строки в обычные, если нужно
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    process_and_save("phishing_websites", df)

print("\n🎉 Готово! Все файлы сохранены в папке 'datasets/'.")
