# CTGAN для табличных датасетов

Набор ноутбуков и скриптов для подготовки табличных датасетов, обучения CTGAN по разным схемам кодирования признаков и оценки качества синтетических выборок через метрики распределений и utility-моделей.

## Что внутри
- `datasets_registry.csv` — реестр датасетов: путь до исходника, путь до служебного `data.csv`, целевая колонка и списки категориальных признаков. Копия файла лежит и в `datasets/datasets_registry.csv`.
- `download_datasets.py` — загрузка исходных CSV из OpenML/UCI, очистка NaN и семплирование до 10k строк.
- `notebooks/dataset_encoding.ipynb` — готовит версии датасетов с `one_hot_encoding`, `label_encoding`, `frequency_encoding` и `original` (LabelEncoder только для таргета). Обновляет `datasets/<dataset>/data.csv`.
- `ctgan_model_creator_new.ipynb` — обучает CTGAN для каждой закодированной версии, сохраняет модели в `datasets/<dataset>/models` и графики лоссов в `training_schedules`, прописывает пути в `data.csv`.
- `card_creator_patched.ipynb` — загружает модели, генерирует синтетику, считает JSD по признакам, utility-gap (LogReg/XGBoost) и формирует сводки `data.csv` и визуальный отчет `report.html`.
- `datasets/` — сырье в `datasets/original/*.csv` и папки по датасетам с закодированными CSV, `data.csv`, моделью и графиком лоссов.
- `requirements.txt` — зависимости (`ctgan`, `sdv`, `sdmetrics`, `pandas`, `scikit-learn`, `xgboost` и др.).

Пример структуры для одного датасета:
```
datasets/
  original/adult.csv
  adult/
    adult_ohe.csv | adult_label.csv | adult_frequency.csv | adult_original.csv
    data.csv  # перечень версий, категориальных колонок, путей до модели/графика
    models/ctgan_adult_<encoding>_model.pkl
    training_schedules/ctgan_adult_<encoding>_losses.png
```

## Как пользоваться
1) Установите окружение
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Скачайте исходные данные (при необходимости обновить `datasets/original`)
```
python3 download_datasets.py
```

3) Подготовьте закодированные версии  
Откройте `notebooks/dataset_encoding.ipynb` и выполните ячейки. Для каждого датасета появятся новые CSV и обновится `datasets/<dataset>/data.csv` с колонками:
- `method` — способ кодирования,
- `path` — путь до подготовленного CSV,
- `New_cat_cols` — итоговые категориальные признаки для CTGAN,
- `model_path` / `schedul_path` — будут заполнены после обучения.

4) Обучите CTGAN  
В `ctgan_model_creator_new.ipynb` вызовите `process_all_datasets(epochs=300, verbose=True)` или `train_single_dataset(<name>, <method>, epochs=...)`. Модели сохранятся в `models/`, графики лоссов — в `training_schedules/`, пути пропишутся в `data.csv`.

5) Посчитайте метрики и сформируйте отчет  
Запустите `card_creator_patched.ipynb`: ноутбук загрузит модели, сэмплирует синтетику, посчитает JSD по признакам, accuracy/R² для LogReg и XGBoost на реальных vs синтетических данных. Результат:
- `data.csv` в корне — агрегированная таблица по всем (датасет, кодировка) с метриками и разницей качеств;
- `report.html` — интерактивный обзор с матрицей кодировок и карточками моделей.

## Добавление нового датасета
1) Поместите исходный CSV в `datasets/original/`.  
2) Добавьте строку в `datasets_registry.csv`: `dataset_name`, `dataset_path` (до исходника), `dataset_csv` (куда писать `data.csv`), `target`, `cat_col` (список категориальных признаков).  
3) Прогоните шаги 3–5 из раздела «Как пользоваться».

## Готовые артефакты
- `data.csv` — сводка метрик для всех обученных моделей на момент последнего прогона.
- `report.html` — статичный HTML-отчет с карточками моделей и графиками лоссов (подтягиваются в base64).
