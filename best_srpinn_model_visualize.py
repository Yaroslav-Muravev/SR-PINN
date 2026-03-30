import os
import glob
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

from main import *  # предполагается, что main.py содержит все необходимые классы


# ====================== Настройки ======================
data_dir = "./files/"
model_path = "best_srpinn_model_voltage.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIS_FIELD_MODE = "abs"       # "real", "imag", "abs"
USE_LOG_SCALE = True
EPS = 1e-12


# ====================== Вспомогательные функции ======================
def parse_complex(s):
    if pd.isna(s):
        return complex(np.nan, np.nan)
    s = str(s).strip().replace(" ", "")
    try:
        return complex(s)
    except Exception:
        return complex(np.nan, np.nan)


def select_component(phi_complex: np.ndarray, mode: str = "abs") -> np.ndarray:
    if mode == "real":
        return np.real(phi_complex)
    if mode == "imag":
        return np.imag(phi_complex)
    if mode == "abs":
        return np.abs(phi_complex)
    raise ValueError(f"Unknown mode={mode!r}. Use 'real', 'imag' or 'abs'.")


def load_coarse_data_for_id(id_: int):
    fname = os.path.join(data_dir, f"pinndata_quick_id_{id_:04d}_coarse.mat")
    if not os.path.exists(fname):
        return None
    with h5py.File(fname, "r") as f:
        X = f["X"][:]
        Y = f["Y"][:]
        Z = f["Z"][:]
        ux = load_complex_from_h5(f, "ux")
        uy = load_complex_from_h5(f, "uy")
        uz = load_complex_from_h5(f, "uz")
        phi = load_complex_from_h5(f, "phi")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    fields = np.stack([
        ux.real.ravel(), ux.imag.ravel(),
        uy.real.ravel(), uy.imag.ravel(),
        uz.real.ravel(), uz.imag.ravel(),
        phi.real.ravel(), phi.imag.ravel()
    ], axis=1)
    tree = cKDTree(coords)
    return coords, fields, tree


def get_batch_for_id(dataset, id_):
    """Возвращает батч (coords, shape, patch) для всех точек одного ID."""
    slc = dataset.get_id_slice(id_)
    if slc.stop - slc.start == 0:
        return None
    # Собираем списки тензоров
    items = [dataset[i] for i in range(slc.start, slc.stop)]
    coords = torch.cat([item["coords"].unsqueeze(0) for item in items], dim=0).to(device)
    shape  = torch.cat([item["shape_params"].unsqueeze(0) for item in items], dim=0).to(device)
    patch  = torch.cat([item["coarse_patch"].unsqueeze(0) for item in items], dim=0).to(device)
    return coords, shape, patch


def predict_fields(model, coords, shape, patch, fields_mean, fields_std):
    with torch.no_grad():
        pred = model(coords, shape, patch).detach().cpu().numpy()
    return pred * fields_std + fields_mean


def get_true_voltage_from_dataset(dataset, id_):
    row = dataset.df[dataset.df["id"] == id_].iloc[0]
    return row["voltage_complex"]


# ====================== Загрузка модели и статистик ======================
# Загружаем чекпоинт
checkpoint = torch.load(model_path, map_location=device)

# ====================== Загрузка модели ======================
model = SRPINN(
    n_spatial=3,
    n_shape_params=2,
    n_coarse_nodes=8,
    n_field_vars=8,
    hidden_dim=256,
    n_blocks=8,
)
# Загружаем веса (прямо state_dict, без обёртки)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ====================== Вычисление глобальных статистик полей по coarse-данным ======================
def compute_global_field_stats(data_dir, coarse_ids):
    all_fields = []
    for id_ in coarse_ids:
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_coarse.mat')
        if not os.path.exists(fname):
            continue
        with h5py.File(fname, 'r') as f:
            ux = load_complex_from_h5(f, 'ux')
            uy = load_complex_from_h5(f, 'uy')
            uz = load_complex_from_h5(f, 'uz')
            phi = load_complex_from_h5(f, 'phi')
        fields = np.stack([
            ux.real.ravel(), ux.imag.ravel(),
            uy.real.ravel(), uy.imag.ravel(),
            uz.real.ravel(), uz.imag.ravel(),
            phi.real.ravel(), phi.imag.ravel()
        ], axis=1)
        all_fields.append(fields)
    if not all_fields:
        raise ValueError("Нет coarse-данных для вычисления статистик")
    all_fields_stack = np.vstack(all_fields)
    fields_mean = all_fields_stack.mean(axis=0)
    fields_std = all_fields_stack.std(axis=0)
    fields_std[fields_std < 1e-8] = 1.0
    return fields_mean, fields_std

coarse_ids = list(range(1, 101))
fields_mean, fields_std = compute_global_field_stats(data_dir, coarse_ids)
print(f"Global fields stats: mean[6]={fields_mean[6]:.3e}, std[6]={fields_std[6]:.3e}")

# ====================== Подготовка данных ======================
fine_files = glob.glob(os.path.join(data_dir, "pinndata_quick_id_*_fine.mat"))
fine_ids = sorted({int(f.split("_")[-2]) for f in fine_files})
print(f"Найдено fine файлов для ID: {fine_ids}")

# Загружаем CSV
fine_df_list = []
for csv_file in glob.glob(os.path.join(data_dir, "results_fine*.csv")):
    fine_df_list.append(pd.read_csv(csv_file))
if not fine_df_list:
    raise FileNotFoundError("Не найдены CSV файлы results_fine*.csv в папке ./files/")
fine_df = pd.concat(fine_df_list, ignore_index=True)
fine_df["voltage_complex"] = fine_df["voltage"].apply(parse_complex)
fine_df = fine_df[fine_df["id"].isin(fine_ids)].copy()

# Создаём датасет с внешними статистиками (теми, что были при обучении)
dataset = CylinderStressDataset(
    data_dir=data_dir,
    csv_df=fine_df,
    ids=fine_ids,
    mesh_type="fine",
    n_neighbors=8,
    normalize=True,
    external_stats=(fields_mean, fields_std)
)

# Подгружаем coarse-данные
all_coarse = {}
for id_ in range(1, 101):
    coarse = load_coarse_data_for_id(id_)
    if coarse is not None:
        all_coarse[id_] = coarse

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

# ====================== Оценка напряжения для всех ID ======================
def evaluate_voltage(model, dataset, device):
    """Возвращает (V_pred, V_true, errors) для всех ID."""
    model.eval()
    ids_unique = sorted(dataset.df["id"].unique())
    V_preds = []
    V_trues = []
    errors = []

    with torch.no_grad():
        for id_ in ids_unique:
            batch = get_batch_for_id(dataset, id_)
            if batch is None:
                continue
            coords, shape, patch = batch
            pred_fields = predict_fields(model, coords, shape, patch,
                                         dataset.fields_mean, dataset.fields_std)
            phi_pred = pred_fields[:, 6] + 1j * pred_fields[:, 7]

            id_idx = dataset.id_to_index[id_]
            bottom_mask = dataset.bottom_mask_list[id_idx]
            top_mask = dataset.top_mask_list[id_idx]
            if not np.any(bottom_mask) or not np.any(top_mask):
                continue

            phi_bottom = phi_pred[bottom_mask].mean()
            phi_top = phi_pred[top_mask].mean()
            V_pred = phi_top - phi_bottom
            V_true = get_true_voltage_from_dataset(dataset, id_)

            V_preds.append(V_pred)
            V_trues.append(V_true)
            errors.append(abs(V_pred - V_true) / (abs(V_true) + EPS))

    return np.array(V_preds), np.array(V_trues), np.array(errors)


V_preds, V_trues, errors = evaluate_voltage(model, dataset, device)

print(f"Средняя относительная ошибка напряжения: {np.mean(errors):.4f}")
print(f"Медианная относительная ошибка: {np.median(errors):.4f}")


# ====================== График scatter + регрессия ======================
mask = np.isfinite(V_trues) & np.isfinite(V_preds)
V_trues_finite = V_trues[mask]
V_preds_finite = V_preds[mask]

V_trues_real = V_trues_finite.real
V_preds_real = V_preds_finite.real

reg = LinearRegression(fit_intercept=True)
reg.fit(V_trues_real.reshape(-1, 1), V_preds_real.reshape(-1, 1))
a = reg.coef_[0][0]
b = reg.intercept_[0]
r2 = reg.score(V_trues_real.reshape(-1, 1), V_preds_real.reshape(-1, 1))

x_line = np.linspace(V_trues_real.min(), V_trues_real.max(), 100)
y_line = a * x_line + b

plt.figure(figsize=(8, 6))
plt.scatter(V_trues_real, V_preds_real, alpha=0.7, label="Данные")
plt.plot(x_line, y_line, "g-", linewidth=2, label=f"МНК: V_pred = {a:.3f}·V_true + {b:.3e}")
plt.plot([V_trues_real.min(), V_trues_real.max()],
         [V_trues_real.min(), V_trues_real.max()], "r--", label="Идеал (y=x)")
plt.xlabel("Истинное напряжение (действительная часть)")
plt.ylabel("Предсказанное напряжение (действительная часть)")
plt.title("Сравнение напряжения на торцах")
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, f"$R^2 = {r2:.4f}$", transform=plt.gca().transAxes,
         verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.savefig("voltage_scatter_with_lr.png", dpi=150, bbox_inches="tight")
plt.show()

# Гистограмма ошибок
plt.figure(figsize=(8, 6))
plt.hist(errors[mask], bins=30, edgecolor="black")
plt.xlabel("Относительная ошибка |V_pred - V_true|/|V_true|")
plt.ylabel("Количество ID")
plt.title("Распределение ошибок напряжения")
plt.grid(True)
plt.savefig("voltage_errors_hist.png", dpi=150, bbox_inches="tight")
plt.show()


# ====================== Ошибка в пространстве параметров ======================
# Получаем r_um, h_um для каждого ID (сохраняем порядок, совпадающий с V_preds)
r_list = []
h_list = []
for id_ in sorted(dataset.df["id"].unique()):
    id_idx = dataset.id_to_index[id_]
    r_list.append(dataset.shape_params[id_idx][0])
    h_list.append(dataset.shape_params[id_idx][1])

r_arr = np.array(r_list)
h_arr = np.array(h_list)

plt.figure(figsize=(10, 6))
sc = plt.scatter(r_arr, h_arr, c=errors, cmap="viridis", s=100, edgecolors="k")
plt.colorbar(sc, label="Относительная ошибка напряжения")
plt.xlabel("Радиус (мкм)")
plt.ylabel("Высота (мкм)")
plt.title("Ошибка предсказания напряжения в пространстве параметров")
plt.grid(True)
plt.savefig("voltage_error_vs_params.png", dpi=150, bbox_inches="tight")
plt.show()


# ====================== Визуализация полей ======================
def visualize_fields_improved(
    id_,
    model,
    device,
    dataset,
    vis_mode="abs",
    use_log_scale=True,
    rel_err_eps=1e-12,
    save_fig=True,
):
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_fine.mat')
    with h5py.File(fname, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:]
        phi = load_complex_from_h5(f, 'phi')

    # Истинное поле в выбранной компоненте
    if vis_mode == "real":
        phi_true_all = np.real(phi)
        phi_label = "Re(phi)"
    elif vis_mode == "imag":
        phi_true_all = np.imag(phi)
        phi_label = "Im(phi)"
    else:
        phi_true_all = np.abs(phi)
        phi_label = "|phi|"

    # Предсказание
    batch = get_batch_for_id(dataset, id_)
    if batch is None:
        print(f"ID {id_}: нет данных в датасете")
        return
    coords, shape, patch = batch
    pred_fields = predict_fields(model, coords, shape, patch,
                                 dataset.fields_mean, dataset.fields_std)
    phi_pred_complex_flat = pred_fields[:, 6] + 1j * pred_fields[:, 7]

    if vis_mode == "real":
        phi_pred_all = phi_pred_complex_flat.real
    elif vis_mode == "imag":
        phi_pred_all = phi_pred_complex_flat.imag
    else:
        phi_pred_all = np.abs(phi_pred_complex_flat)

    phi_pred_all = phi_pred_all.reshape(X.shape)

    # Выбираем слой с максимальной средней амплитудой
    z_means = np.mean(phi_true_all, axis=(0, 1))
    idx_z = np.argmax(z_means)
    z_val = Z[0, 0, idx_z]
    print(f"ID {id_}: выбран слой z={z_val:.3e} м, среднее |field|={z_means[idx_z]:.3e}")

    phi_true_layer = phi_true_all[:, :, idx_z]
    phi_pred_layer = phi_pred_all[:, :, idx_z]
    X_layer = X[:, :, idx_z]
    Y_layer = Y[:, :, idx_z]

    x_flat = X_layer.ravel()
    y_flat = Y_layer.ravel()
    true_flat = phi_true_layer.ravel()
    pred_flat = phi_pred_layer.ravel()

    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(true_flat) & np.isfinite(pred_flat)
    if valid.sum() < 10:
        print(f"  Недостаточно точек для визуализации (valid={valid.sum()})")
        return

    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    true_flat = true_flat[valid]
    pred_flat = pred_flat[valid]
    tri = Triangulation(x_flat, y_flat)

    eps_plot = 1e-30
    if use_log_scale:
        true_plot = np.log10(np.abs(true_flat) + eps_plot)
        pred_plot = np.log10(np.abs(pred_flat) + eps_plot)
        field_name = f"log10({phi_label})"
    else:
        true_plot = true_flat
        pred_plot = pred_flat
        field_name = phi_label

    abs_err = np.abs(true_flat - pred_flat)

    # Относительная ошибка
    rel_err = np.full_like(abs_err, np.nan)
    mask_large = np.abs(true_flat) > rel_err_eps
    rel_err[mask_large] = abs_err[mask_large] / np.abs(true_flat[mask_large])
    mask_rel = np.isfinite(rel_err)
    if mask_rel.sum() < 10:
        rel_tri = None
        rel_plot = None
    else:
        rel_tri = Triangulation(x_flat[mask_rel], y_flat[mask_rel])
        rel_plot = rel_err[mask_rel]

    vmin_field = np.nanmin([np.nanmin(true_plot), np.nanmin(pred_plot)])
    vmax_field = np.nanmax([np.nanmax(true_plot), np.nanmax(pred_plot)])
    err_vmax = np.nanpercentile(abs_err, 95)
    if not np.isfinite(err_vmax) or err_vmax <= 0:
        err_vmax = np.nanmax(abs_err) if np.nanmax(abs_err) > 0 else 1.0

    if rel_tri is not None:
        rel_vmax = np.nanpercentile(rel_plot, 95)
        if not np.isfinite(rel_vmax) or rel_vmax <= 0:
            rel_vmax = np.nanmax(rel_plot) if np.nanmax(rel_plot) > 0 else 1.0

    # Рисование
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    t1 = ax1.tricontourf(tri, true_plot, levels=60, cmap='viridis',
                         vmin=vmin_field, vmax=vmax_field)
    ax1.set_title(f'ID {id_}: истинное {field_name}, z={z_val:.3e} м')
    ax1.set_aspect('equal')
    plt.colorbar(t1, ax=ax1, label=field_name)

    t2 = ax2.tricontourf(tri, pred_plot, levels=60, cmap='viridis',
                         vmin=vmin_field, vmax=vmax_field)
    ax2.set_title(f'ID {id_}: предсказанное {field_name}')
    ax2.set_aspect('equal')
    plt.colorbar(t2, ax=ax2, label=field_name)

    t3 = ax3.tricontourf(tri, abs_err, levels=60, cmap='magma',
                         vmin=0, vmax=err_vmax)
    ax3.set_title('Абсолютная ошибка |true-pred|')
    ax3.set_aspect('equal')
    plt.colorbar(t3, ax=ax3, label='ошибка')

    if rel_tri is not None:
        t4 = ax4.tricontourf(rel_tri, rel_plot, levels=60, cmap='magma',
                             vmin=0, vmax=rel_vmax)
        ax4.set_title('Относительная ошибка')
        plt.colorbar(t4, ax=ax4, label='отн. ошибка')
    else:
        ax4.text(0.5, 0.5, 'Недостаточно данных\nдля отображения\nотносительной ошибки',
                 transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Относительная ошибка')
    ax4.set_aspect('equal')

    plt.tight_layout()
    if save_fig:
        plt.savefig(f'phi_comparison_id_{id_}_{vis_mode}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Визуализация для первых трёх ID
for id_ in fine_ids[:3]:
    visualize_fields_improved(
        id_, model, device, dataset,
        vis_mode=VIS_FIELD_MODE,
        use_log_scale=USE_LOG_SCALE,
        save_fig=True
    )