"""
Скрипт для автоматической тренировки CTGAN моделей для всех датасетов.
Использует структуру данных из datasets_registry.csv и data.csv.
"""

import pandas as pd
import numpy as np
from ctgan import CTGAN
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import ast
import os
import argparse


def load_datasets_registry(registry_path='datasets/datasets_registry.csv'):
    """Загружает реестр датасетов."""
    df = pd.read_csv(registry_path, skipinitialspace=True)

    datasets_list = []
    for _, row in df.iterrows():
        if pd.isna(row['cat_col']):
            cat_cols_list = []
        else:
            cat_col_str = str(row['cat_col']).replace('\n', '').replace('\r', '').strip()
            try:
                if cat_col_str:
                    cat_cols_list = ast.literal_eval(cat_col_str)
                else:
                    cat_cols_list = []
            except (ValueError, SyntaxError):
                cat_cols_list = []

        dataset_info = {
            'dataset_name': row['dataset_name'].strip(),
            'dataset_path': row['dataset_path'].strip(),
            'dataset_csv': row['dataset_csv'].strip(),
            'target': row['target'].strip(),
            'cat_cols': cat_cols_list
        }
        datasets_list.append(dataset_info)

    return datasets_list


def load_encoded_datasets(dataset_info):
    """Загружает информацию о закодированных версиях датасета."""
    data_csv_path = Path(dataset_info['dataset_csv'])

    if not os.path.exists(data_csv_path):
        print(f"  ⚠️ Файл {data_csv_path} не найден")
        return []

    df = pd.read_csv(data_csv_path)

    encoded_datasets = []
    for _, row in df.iterrows():
        try:
            new_cat_cols = ast.literal_eval(str(row['New_cat_cols']))
        except:
            new_cat_cols = []

        encoded_info = {
            'method': row['method'],
            'path': row['path'],
            'New_cat_cols': new_cat_cols,
            'model_path': row.get('model_path', ''),
            'schedul_path': row.get('schedul_path', ''),
            'dataset_name': dataset_info['dataset_name'],
            'dataset_folder': Path(dataset_info['dataset_csv']).parent
        }
        encoded_datasets.append(encoded_info)

    return encoded_datasets


def plot_ctgan_losses(loss_df, smooth_window=10, save_path=None):
    """Создает график лоссов CTGAN."""
    if loss_df is None or len(loss_df) == 0:
        raise ValueError("loss_df пустой")

    cols_lower = {c.lower(): c for c in loss_df.columns}
    g_col = next((cols_lower[c] for c in cols_lower if "gen" in c), None)
    d_col = next((cols_lower[c] for c in cols_lower if "disc" in c), None)

    if g_col is None or d_col is None:
        if len(loss_df.columns) < 2:
            raise ValueError("loss_df должен содержать хотя бы 2 столбца")
        g_col, d_col = loss_df.columns[:2]

    epochs = np.arange(1, len(loss_df) + 1)

    df = loss_df.copy()
    for c in [g_col, d_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g_smooth = df[g_col].rolling(smooth_window, min_periods=1).mean()
    d_smooth = df[d_col].rolling(smooth_window, min_periods=1).mean()

    plt.figure(figsize=(10, 6), dpi=120)

    plt.plot(epochs, df[g_col], alpha=0.25, linewidth=1, label=f"{g_col} (raw)")
    plt.plot(epochs, df[d_col], alpha=0.25, linewidth=1, label=f"{d_col} (raw)")

    plt.plot(epochs, g_smooth, linewidth=2.5, label=f"{g_col} (smoothed)")
    plt.plot(epochs, d_smooth, linewidth=2.5, label=f"{d_col} (smoothed)")

    def annotate_series(y, name):
        y_last = float(y.iloc[-1])
        y_min = float(y.min())
        x_min = int(y.idxmin()) + 1
        plt.scatter([len(y)], [y_last], s=30)
        plt.text(len(y), y_last, f"  last: {y_last:.3f}", va="center")
        plt.scatter([x_min], [y_min], s=30)
        plt.text(x_min, y_min, f"  min@{x_min}: {y_min:.3f}", va="center")

    annotate_series(g_smooth, g_col)
    annotate_series(d_smooth, d_col)

    plt.title("CTGAN Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def train_ctgan_for_encoded_dataset(encoded_info, epochs=300, verbose=True):
    """Тренирует CTGAN модель."""
    try:
        data_path = Path(encoded_info['path'])
        print(f"\n{'='*70}")
        print(f"📊 Датасет: {encoded_info['dataset_name']}")
        print(f"🔧 Метод: {encoded_info['method']}")
        print(f"📁 Путь: {data_path}")

        if not os.path.exists(data_path):
            print(f"  ⚠️ Файл {data_path} не найден, пропускаем")
            return None, None

        df = pd.read_csv(data_path)
        print(f"  ✅ Загружено {len(df)} строк, {len(df.columns)} колонок")

        discrete_features = [col for col in encoded_info['New_cat_cols'] if col in df.columns]
        print(f"  🏷️ Дискретных признаков: {len(discrete_features)}")

        print(f"  🚀 Начинаем обучение CTGAN ({epochs} эпох)...")
        ctgan = CTGAN(epochs=epochs, verbose=verbose)
        ctgan.fit(df, discrete_features)

        loss_df = ctgan.loss_values
        print(f"  ✅ Обучение завершено!")

        return ctgan, loss_df

    except Exception as e:
        print(f"  ❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_ctgan_results(ctgan, loss_df, encoded_info, data_csv_path):
    """Сохраняет модель, график и обновляет data.csv."""
    try:
        dataset_folder = encoded_info['dataset_folder']
        dataset_name = encoded_info['dataset_name']
        method = encoded_info['method']

        models_folder = dataset_folder / 'models'
        schedules_folder = dataset_folder / 'training_schedules'
        models_folder.mkdir(exist_ok=True)
        schedules_folder.mkdir(exist_ok=True)

        model_filename = f"ctgan_{dataset_name}_{method}_model.pkl"
        schedule_filename = f"ctgan_{dataset_name}_{method}_losses.png"

        model_path = models_folder / model_filename
        schedule_path = schedules_folder / schedule_filename

        ctgan.save(str(model_path))
        print(f"  💾 Модель сохранена: {model_path}")

        plot_ctgan_losses(loss_df, save_path=str(schedule_path))
        print(f"  📈 График сохранен: {schedule_path}")

        df = pd.read_csv(data_csv_path)
        mask = df['method'] == method

        df.loc[mask, 'model_path'] = str(model_path)
        df.loc[mask, 'schedul_path'] = str(schedule_path)

        df.to_csv(data_csv_path, index=False)
        print(f"  📝 Обновлен {data_csv_path}")

    except Exception as e:
        print(f"  ❌ Ошибка при сохранении: {e}")
        import traceback
        traceback.print_exc()


def process_all_datasets(epochs=300, verbose=True, dataset_filter=None, method_filter=None):
    """Обрабатывает все датасеты из реестра."""
    print("="*70)
    print("🚀 CTGAN Model Creator")
    print("="*70)

    datasets = load_datasets_registry()
    print(f"\n📋 Загружено датасетов: {len(datasets)}")

    # Применяем фильтр по датасету
    if dataset_filter:
        datasets = [ds for ds in datasets if ds['dataset_name'] == dataset_filter]
        if not datasets:
            print(f"❌ Датасет '{dataset_filter}' не найден")
            return
        print(f"🔍 Фильтр: только датасет '{dataset_filter}'")

    total_models = 0
    successful_models = 0

    for dataset_info in datasets:
        print(f"\n{'='*70}")
        print(f"📦 Обработка датасета: {dataset_info['dataset_name']}")
        print(f"{'='*70}")

        encoded_datasets = load_encoded_datasets(dataset_info)
        print(f"  📊 Найдено закодированных версий: {len(encoded_datasets)}")

        if len(encoded_datasets) == 0:
            print(f"  ⚠️ Пропускаем датасет (нет закодированных версий)")
            continue

        # Применяем фильтр по методу
        if method_filter:
            encoded_datasets = [enc for enc in encoded_datasets if enc['method'] == method_filter]
            if not encoded_datasets:
                print(f"  ⚠️ Метод '{method_filter}' не найден, пропускаем")
                continue
            print(f"  🔍 Фильтр: только метод '{method_filter}'")

        for encoded_info in encoded_datasets:
            total_models += 1

            ctgan, loss_df = train_ctgan_for_encoded_dataset(
                encoded_info,
                epochs=epochs,
                verbose=verbose
            )

            if ctgan is not None and loss_df is not None:
                data_csv_path = Path(dataset_info['dataset_csv'])
                save_ctgan_results(ctgan, loss_df, encoded_info, data_csv_path)
                successful_models += 1

    print(f"\n{'='*70}")
    print(f"✅ ЗАВЕРШЕНО!")
    print(f"{'='*70}")
    print(f"  Всего моделей: {total_models}")
    print(f"  Успешно обучено: {successful_models}")
    print(f"  Ошибок: {total_models - successful_models}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Тренировка CTGAN моделей для датасетов')
    parser.add_argument('--epochs', type=int, default=300, help='Количество эпох обучения')
    parser.add_argument('--dataset', type=str, default=None, help='Имя конкретного датасета (опционально)')
    parser.add_argument('--method', type=str, default=None, help='Метод кодирования (опционально)')
    parser.add_argument('--quiet', action='store_true', help='Не выводить прогресс обучения')

    args = parser.parse_args()

    process_all_datasets(
        epochs=args.epochs,
        verbose=not args.quiet,
        dataset_filter=args.dataset,
        method_filter=args.method
    )

