import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import h5py
import os
import glob
from typing import List, Tuple
from collections import defaultdict
from main import *

# Импортируем классы из вашего обучающего кода (или скопируем их сюда)
# Здесь предполагается, что классы SRPINN, CylinderStressDataset, load_complex_from_h5 и т.д. уже определены.
# Если вы переиспользуете тот же скрипт, то они уже загружены.
# В противном случае скопируйте определения сюда (лучше вынести в отдельный модуль).

# ---------------------- Настройки ----------------------
data_dir = "./files/"
model_path = "best_srpinn_model_voltage.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Загрузка модели ----------------------
model = SRPINN(
    n_spatial=3,
    n_shape_params=2,
    n_coarse_nodes=8,
    n_field_vars=8,
    hidden_dim=256,
    n_blocks=8
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ---------------------- Подготовка данных: все fine ID ----------------------
# Находим все fine .mat файлы
fine_files = glob.glob(os.path.join(data_dir, "pinndata_quick_id_*_fine.mat"))
fine_ids = [int(f.split('_')[-2]) for f in fine_files]
fine_ids = sorted(fine_ids)
print(f"Найдено fine файлов для ID: {fine_ids}")

# Загружаем объединённый CSV с fine результатами
fine_df_list = []
for csv_file in glob.glob(os.path.join(data_dir, "results_fine*.csv")):
    fine_df_list.append(pd.read_csv(csv_file))
fine_df = pd.concat(fine_df_list, ignore_index=True)


# Парсим комплексное напряжение
def parse_complex(s):
    s = s.strip().replace(' ', '')
    # упрощённо: попробуем комплекс()
    try:
        return complex(s)
    except:
        return complex(np.nan, np.nan)


fine_df['voltage_complex'] = fine_df['voltage'].apply(parse_complex)


# Для каждого ID нужно будет загрузить coarse данные (для построения патчей)
# Мы не будем создавать полный датасет, а реализуем функцию предсказания для одного ID.

def load_coarse_data_for_id(id_: int):
    """Загружает coarse сетку и поля для заданного ID."""
    fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_coarse.mat')
    if not os.path.exists(fname):
        return None
    with h5py.File(fname, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:]
        ux = load_complex_from_h5(f, 'ux')
        uy = load_complex_from_h5(f, 'uy')
        uz = load_complex_from_h5(f, 'uz')
        phi = load_complex_from_h5(f, 'phi')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    fields = np.stack([
        ux.real.ravel(), ux.imag.ravel(),
        uy.real.ravel(), uy.imag.ravel(),
        uz.real.ravel(), uz.imag.ravel(),
        phi.real.ravel(), phi.imag.ravel()
    ], axis=1)
    # Строим kd-tree для быстрого поиска
    tree = cKDTree(coords)
    return coords, fields, tree


def load_fine_data_for_id(id_: int):
    """Загружает fine сетку и поля для заданного ID."""
    fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_fine.mat')
    if not os.path.exists(fname):
        return None
    with h5py.File(fname, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:]
        ux = load_complex_from_h5(f, 'ux')
        uy = load_complex_from_h5(f, 'uy')
        uz = load_complex_from_h5(f, 'uz')
        phi = load_complex_from_h5(f, 'phi')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    fields = np.stack([
        ux.real.ravel(), ux.imag.ravel(),
        uy.real.ravel(), uy.imag.ravel(),
        uz.real.ravel(), uz.imag.ravel(),
        phi.real.ravel(), phi.imag.ravel()
    ], axis=1)
    return coords, fields, X.shape  # также возвращаем исходную форму для reshape


def predict_fields_for_id(model, id_, coarse_tree, coarse_coords, coarse_fields,
                          fine_coords, batch_size=1024, normalize_stats=None):
    """
    Предсказывает поля для всех точек fine_coords.
    normalize_stats: словарь с ключами 'coords_mean', 'coords_std', 'shape_mean', 'shape_std',
                     'fields_mean', 'fields_std' – те же, что использовались при обучении.
    Если None, то нормализация не применяется (тогда нужно предобучить статистики).
    """
    # Параметры формы (нужно взять из CSV)
    # Для простоты передадим их отдельно или получим из fine_df.
    # Здесь будем передавать shape_params как константу для данного ID.
    # В реальности shape_params должны быть те же, что при обучении.
    # Допустим, мы их знаем из CSV.
    pass


# Упростим: будем использовать датасет, как при обучении, но без перемешивания и с загрузкой всех точек.
# Для этого создадим временный датасет, который загрузит fine точки и для них построит патчи.
# Однако проще всего взять уже готовый класс CylinderStressDataset, но с mesh_type='fine' и нормализацией.
# Но для этого нужно передать статистики нормализации. Можно их сохранить при обучении.
# В нашем обучающем коде статистики были сохранены в переменной stats. Давайте их сохраним в файл.
# Для простоты предположим, что мы их загрузим из файла stats.npy или просто продублируем вычисление.
# Но чтобы не усложнять, пересчитаем статистики заново на всех fine данных (они будут очень близки к обучающим).

# Создадим полный fine датасет без разбиения (только для инференса)
# Используем тот же класс CylinderStressDataset, но с normalize=True.
# Это автоматически пересчитает статистики по всем fine данным, что может немного отличаться от обучающих,
# но для визуализации сойдёт.

# Однако для корректного предсказания нам нужны coarse деревья для каждого ID.
# Поэтому мы загрузим coarse данные для всех fine ID и передадим в датасет.

# Создадим датасет, который будет содержать все fine точки.
# Но чтобы не загружать всё в память, будем использовать DataLoader с batch_size.

# Реализуем инференс напрямую:

def compute_voltage_for_all_ids(model, fine_ids, fine_df, device, batch_size=4096):
    results = []
    model.eval()
    # Для каждого ID загружаем coarse и fine данные
    for id_ in fine_ids:
        print(f"Processing ID {id_}...")
        coarse = load_coarse_data_for_id(id_)
        if coarse is None:
            continue
        coarse_coords, coarse_fields, coarse_tree = coarse
        fine_data = load_fine_data_for_id(id_)
        if fine_data is None:
            continue
        fine_coords, fine_fields, fine_shape = fine_data
        N = fine_coords.shape[0]

        # Параметры формы из fine_df
        row = fine_df[fine_df['id'] == id_].iloc[0]
        r_um = row['r_um']
        h_um = row['h_um']
        shape = np.array([r_um, h_um])
        V_true = row['voltage_complex']

        # Нормализация: нужны статистики. Пересчитаем их на всех fine данных?
        # Но у нас нет глобальных статистик. Лучше использовать те, что были при обучении.
        # Для простоты пока отключим нормализацию, но тогда сеть не сработает.
        # Вместо этого загрузим статистики из файла, если сохранили.
        # Здесь я предположу, что мы сохранили их в stats.npy.
        # Если нет, нужно либо пересчитать по обучающей выборке и сохранить.
        # Для примера, возьмём заготовку:
        stats = np.load('training_stats.npz')  # должен содержать coords_mean, coords_std, ...
        coords_mean = stats['coords_mean']
        coords_std = stats['coords_std']
        shape_mean = stats['shape_mean']
        shape_std = stats['shape_std']
        fields_mean = stats['fields_mean']
        fields_std = stats['fields_std']

        # Предсказываем поля по батчам
        pred_fields_denorm = np.zeros((N, 8))
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_coords = fine_coords[start:end]
            # Нормализуем координаты и shape
            batch_coords_norm = (batch_coords - coords_mean) / coords_std
            batch_shape_norm = (shape - shape_mean) / shape_std
            batch_shape_norm = np.tile(batch_shape_norm, (end - start, 1))
            # Для каждой точки нужен coarse-патч
            patches = []
            for pt in batch_coords_norm:
                # Но патч строится в исходных координатах? Внимание: coarse координаты тоже должны быть нормализованы.
                # Лучше строить патч в исходных координатах, потом нормализовать.
                # Упростим: будем строить патч в исходных координатах, потом нормализовать поля внутри патча.
                # Это слишком громоздко. Проще использовать датасет, который уже делает всё правильно.
                pass
        # Вместо ручного цикла лучше создать датасет.
        break
    return results


# Более простой путь: создать датасет для всех fine ID и использовать DataLoader.
# Для этого нужно, чтобы датасет умел загружать coarse данные для каждого ID.
# Мы уже реализовали CylinderStressDataset с set_coarse_data.
# Создадим датасет, который включает все fine ID, и передадим ему coarse деревья для всех ID.

# Сначала соберём coarse деревья для всех ID (1..100)
all_coarse = {}
for id_ in range(1, 101):
    coarse = load_coarse_data_for_id(id_)
    if coarse:
        all_coarse[id_] = coarse

# Создадим DataFrame для всех fine данных
fine_df_all = fine_df[fine_df['id'].isin(fine_ids)].copy()
# Создадим датасет
dataset = CylinderStressDataset(
    data_dir=data_dir,
    csv_df=fine_df_all,
    ids=fine_ids,
    mesh_type='fine',
    n_neighbors=8,
    normalize=True
)
# Передаём coarse данные
coarse_coords_list = []
coarse_fields_list = []
coarse_ids_list = []
for id_ in fine_ids:
    if id_ in all_coarse:
        coords, fields, _ = all_coarse[id_]
        coarse_coords_list.append(coords)
        coarse_fields_list.append(fields)
        coarse_ids_list.append(id_)
dataset.set_coarse_data(coarse_coords_list, coarse_fields_list, coarse_ids_list)

# Теперь можно сделать DataLoader и пройти по всем точкам
loader = DataLoader(dataset, batch_size=4096, shuffle=False)

pred_fields_all = []
target_fields_all = []
ids_all = []
with torch.no_grad():
    for batch in loader:
        coords = batch['coords'].to(device)
        shape = batch['shape_params'].to(device)
        patch = batch['coarse_patch'].to(device)
        pred = model(coords, shape, patch).cpu().numpy()
        target = batch['target'].numpy()
        ids = batch['id'].numpy()
        pred_fields_all.append(pred)
        target_fields_all.append(target)
        ids_all.extend(ids)
pred_fields_all = np.vstack(pred_fields_all)
target_fields_all = np.vstack(target_fields_all)

# Денормализация
fields_mean = dataset.fields_mean
fields_std = dataset.fields_std
pred_fields_denorm = pred_fields_all * fields_std + fields_mean
target_fields_denorm = target_fields_all * fields_std + fields_mean

# Теперь для каждого ID нужно восстановить маски торцов и вычислить напряжение.
# У нас есть все предсказания, но они перемешаны. Удобнее сгруппировать по ID.
# Для этого создадим словарь: id -> список индексов в предсказаниях.
id_to_indices = defaultdict(list)
for i, id_ in enumerate(ids_all):
    id_to_indices[id_].append(i)

voltage_errors = []
V_pred_list = []
V_true_list = []
for id_ in fine_ids:
    indices = id_to_indices[id_]
    # Получаем phi_re (индекс 6) для этих индексов
    phi_re_pred = pred_fields_denorm[indices, 6]
    phi_re_true = target_fields_denorm[indices, 6]
    # Получаем маски торцов из датасета
    id_idx = dataset.id_to_index[id_]
    bottom_mask = dataset.bottom_mask_list[id_idx]
    top_mask = dataset.top_mask_list[id_idx]
    # В датасете маски хранятся в порядке исходных точек, но наши предсказания могут быть в другом порядке,
    # потому что DataLoader перемешивает. Нам нужно синхронизировать маски с порядком предсказаний.
    # Проще: в датасете есть свойство coords_list, которое хранит исходный порядок.
    # Мы можем получить slice для этого ID и сравнить.
    # Лучше не полагаться на перемешивание, а создать отдельный датасет без shuffle для каждого ID.
    # Упростим: будем вычислять напряжение на основе всех точек ID, но с использованием масок,
    # если порядок не сохранён, это ошибётся.
    # Поэтому правильнее создать отдельный датасет для каждого ID без shuffle.
    pass

# Вместо этого мы можем воспользоваться функцией compute_voltage_error, которая уже написана в обучающем коде,
# но она ожидает val_dataset. Создадим val_dataset из всех fine ID и используем её.

val_dataset = CylinderStressDataset(
    data_dir=data_dir,
    csv_df=fine_df_all,
    ids=fine_ids,
    mesh_type='fine',
    n_neighbors=8,
    normalize=True
)
val_dataset.set_coarse_data(coarse_coords_list, coarse_fields_list, coarse_ids_list)
# Вычислим ошибку напряжения с выводом подробностей
#from train_srpinn import compute_voltage_error  # если функция определена

voltage_error = compute_voltage_error(model, val_dataset, device, verbose=True)
print(f"Средняя относительная ошибка напряжения на всех fine ID: {voltage_error:.4f}")


# Теперь построим графики: scatter predicted vs true, гистограмма ошибок.
# Для этого нужно собрать V_pred и V_true для каждого ID.
def collect_voltages(model, dataset, device):
    model.eval()
    ids_unique = dataset.df['id'].values
    V_preds = []
    V_trues = []
    for id_ in ids_unique:
        slc = dataset.get_id_slice(id_)
        if slc.stop - slc.start == 0:
            continue
        indices = list(range(slc.start, slc.stop))
        # Загружаем данные для этого ID в правильном порядке (не перемешивая)
        batch_coords = []
        batch_shape = []
        batch_patch = []
        for idx in indices:
            item = dataset[idx]
            batch_coords.append(item['coords'].unsqueeze(0))
            batch_shape.append(item['shape_params'].unsqueeze(0))
            batch_patch.append(item['coarse_patch'].unsqueeze(0))
        coords = torch.cat(batch_coords, dim=0).to(device)
        shape = torch.cat(batch_shape, dim=0).to(device)
        patch = torch.cat(batch_patch, dim=0).to(device)
        with torch.no_grad():
            pred = model(coords, shape, patch).cpu().numpy()
        # Денормализация
        pred_denorm = pred * dataset.fields_std + dataset.fields_mean
        phi_pred = pred_denorm[:, 6] + 1j * pred_denorm[:, 7]
        # Маски
        id_idx = dataset.id_to_index[id_]
        bottom_mask = dataset.bottom_mask_list[id_idx]
        top_mask = dataset.top_mask_list[id_idx]
        phi_bottom = phi_pred[bottom_mask].mean()
	  phi_top = phi_pred[top_mask].mean()
        V_pred = phi_top - phi_bottom
        V_true = dataset.df[dataset.df['id'] == id_].iloc[0]['voltage_complex']
        V_preds.append(V_pred)
        V_trues.append(V_true)
    return np.array(V_preds), np.array(V_trues)


V_preds, V_trues = collect_voltages(model, val_dataset, device)
relative_errors = np.abs(V_preds - V_trues) / (np.abs(V_trues) + 1e-15)

# График scatter
plt.figure(figsize=(8, 6))
plt.scatter(V_trues, V_preds, alpha=0.7)
plt.plot([V_trues.min(), V_trues.max()], [V_trues.min(), V_trues.max()], 'r--', label='идеал')
plt.xlabel('Истинное напряжение (действ. часть)')
plt.ylabel('Предсказанное напряжение')
plt.title('Сравнение напряжения на торцах')
plt.legend()
plt.grid(True)
plt.savefig('voltage_scatter.png')
plt.show()

# Гистограмма относительных ошибок
plt.figure(figsize=(8, 6))
plt.hist(relative_errors, bins=30, edgecolor='black')
plt.xlabel('Относительная ошибка |V_pred - V_true|/|V_true|')
plt.ylabel('Количество ID')
plt.title('Распределение ошибок напряжения')
plt.grid(True)
plt.savefig('voltage_errors_hist.png')
plt.show()

# Также можно построить boxplot ошибок по ID (если есть повторяющиеся измерения)

def visualize_fields_for_id(id_, model, device, dataset, data_dir):
    fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_fine.mat')
    with h5py.File(fname, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:]
        phi = load_complex_from_h5(f, 'phi')

    phi_true = phi.real

    # Берём все точки этого ID в правильном порядке
    slc = dataset.get_id_slice(id_)
    indices = list(range(slc.start, slc.stop))

    batch_coords = []
    batch_shape = []
    batch_patch = []
    for idx in indices:
        item = dataset[idx]
        batch_coords.append(item['coords'].unsqueeze(0))
        batch_shape.append(item['shape_params'].unsqueeze(0))
        batch_patch.append(item['coarse_patch'].unsqueeze(0))

    coords = torch.cat(batch_coords, dim=0).to(device)
    shape = torch.cat(batch_shape, dim=0).to(device)
    patch = torch.cat(batch_patch, dim=0).to(device)

    with torch.no_grad():
        pred = model(coords, shape, patch).cpu().numpy()

    pred_denorm = pred * dataset.fields_std + dataset.fields_mean
    phi_pred_flat = pred_denorm[:, 6]
    phi_pred = phi_pred_flat.reshape(X.shape)

    # Верхний торец
    phi_true_top = phi_true[:, :, -1]
    phi_pred_top = phi_pred[:, :, -1]
    X_top = X[:, :, -1]
    Y_top = Y[:, :, -1]

    # Относительная ошибка
    eps = 1e-12
    rel_error = np.abs(phi_true_top - phi_pred_top) / np.maximum(np.abs(phi_true_top), eps)

    # Для scatter нужен плоский вид
    x = X_top.ravel()
    y = Y_top.ravel()
    true_vals = phi_true_top.ravel()
    pred_vals = phi_pred_top.ravel()
    rel_vals = rel_error.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    s1 = axes[0].scatter(x, y, c=true_vals, s=6, cmap='RdBu_r')
    axes[0].set_title(f'ID {id_}: истинное phi')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(s1, ax=axes[0], label='phi')

    s2 = axes[1].scatter(x, y, c=pred_vals, s=6, cmap='RdBu_r')
    axes[1].set_title(f'ID {id_}: предсказанное phi')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(s2, ax=axes[1], label='phi')

    s3 = axes[2].scatter(x, y, c=rel_vals, s=6, cmap='hot', vmin=0.0, vmax=1.0)
    axes[2].set_title('Относительная ошибка |true - pred| / max(|true|, eps)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(s3, ax=axes[2], label='relative error')

    plt.tight_layout()
    plt.savefig(f'phi_comparison_relative_id_{id_}.png', dpi=150)
    plt.show(block=True)
    plt.close(fig)

## Визуализация поля для нескольких ID (например, первые 3)
##for id_ in fine_ids[:3]:
##    # Загружаем fine сетку и поле phi из .mat
##    fine_data = load_fine_data_for_id(id_)
##    if fine_data is None:
##        continue
##    fine_coords, fine_fields, shape3d = fine_data
##    phi_true = fine_fields[:, 6]  # действительная часть phi
##    # Восстанавливаем 3D сетку
##    X_shape = shape3d
##    phi_true_3d = phi_true.reshape(X_shape)
##    # Предсказываем поле для всех точек этого ID
##    # Используем dataset для получения предсказаний в том же порядке, что и исходные точки.
##    # Для этого создадим отдельный датасет для одного ID с normalize=False? Нет, с normalize=True,
##    # но тогда нужно применить обратную нормализацию.
##    # Проще: используем уже имеющийся dataset и вытащим предсказания для этого ID в правильном порядке.
##    slc = val_dataset.get_id_slice(id_)
##    indices = list(range(slc.start, slc.stop))
##    # Порядок индексов в dataset должен соответствовать порядку точек в .mat, если датасет не перемешивался.
##    # В нашем dataset порядок сохраняется, потому что мы не делали shuffle при создании индексов.
##    # Проверим: в CylinderStressDataset точки идут последовательно по ID, и внутри ID – в порядке flatten.
##    # Значит, предсказания для этого ID можно получить в том же порядке.
##    batch_coords = []
##    batch_shape = []
##    batch_patch = []
##    for idx in indices:
##        item = val_dataset[idx]
##        batch_coords.append(item['coords'].unsqueeze(0))
##        batch_shape.append(item['shape_params'].unsqueeze(0))
##        batch_patch.append(item['coarse_patch'].unsqueeze(0))
##    coords = torch.cat(batch_coords, dim=0).to(device)
##    shape = torch.cat(batch_shape, dim=0).to(device)
##    patch = torch.cat(batch_patch, dim=0).to(device)
##    with torch.no_grad():
##        pred = model(coords, shape, patch).cpu().numpy()
##    pred_denorm = pred * val_dataset.fields_std + val_dataset.fields_mean
##    phi_pred = pred_denorm[:, 6]
##    phi_pred_3d = phi_pred.reshape(X_shape)
##
##    # Визуализируем срез (например, на верхнем торце)
##    # Найдём индекс z-слоя, соответствующего верхней грани (z_max)
##    z_vals = fine_coords[:, 2].reshape(X_shape)
##    z_max = z_vals.max()
##    # Ищем слой с максимальным z (с допуском)
##    eps = 1e-6 * (z_max - z_vals.min())
##    top_layer = np.abs(z_vals - z_max) < eps
##    if not np.any(top_layer):
##        # если нет точного совпадения, берём ближайший
##        idx_z = np.argmin(np.abs(z_vals[0, 0, :] - z_max))
##        top_layer = np.zeros(z_vals.shape, dtype=bool)
##        top_layer[:, :, idx_z] = True
##    # Берём срез
##    phi_true_top = phi_true_3d[top_layer]
##    phi_pred_top = phi_pred_3d[top_layer]
##    # Координаты X,Y для этого слоя
##    X_plane = fine_coords[:, 0].reshape(X_shape)[top_layer]
##    Y_plane = fine_coords[:, 1].reshape(X_shape)[top_layer]
##
##    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
##    sc1 = axes[0].scatter(X_plane, Y_plane, c=phi_true_top, cmap='viridis', s=2)
##    axes[0].set_title(f'ID {id_}: Истинное поле phi (верхний торец)')
##    axes[0].set_xlabel('x (м)')
##    axes[0].set_ylabel('y (м)')
##    plt.colorbar(sc1, ax=axes[0], label='phi (В)')
##
##    sc2 = axes[1].scatter(X_plane, Y_plane, c=phi_pred_top, cmap='viridis', s=2)
##    axes[1].set_title(f'ID {id_}: Предсказанное поле phi (верхний торец)')
##    axes[1].set_xlabel('x (м)')
##    axes[1].set_ylabel('y (м)')
##    plt.colorbar(sc2, ax=axes[1], label='phi (В)')
##
##    plt.tight_layout()
##    plt.savefig(f'phi_comparison_id_{id_}.png')
##    plt.show()

for id_ in fine_ids[:3]:
    visualize_fields_for_id(id_, model, device, val_dataset, data_dir)
