import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
import h5py

from src.config import PATH_TO_FILES
from src.model import SRPINN
from src.datasets import CylinderStressDataset, parse_idx_train
from src.data_utils import (
    load_mat_with_cache, load_all_csv_cached, compute_stats_incremental,
    generate_fine_fields, load_complex_from_h5, parse_complex
)

model_path = "best_srpinn_model_with_stats.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIS_FIELD_MODE = "abs"
USE_LOG_SCALE = True
EPS = 1e-12


def select_component(phi_real: np.ndarray, mode: str = "abs") -> np.ndarray:
    """Выбирает компоненту для визуализации (только real или abs)."""
    if mode == "real":
        return phi_real
    if mode == "abs":
        return np.abs(phi_real)
    if mode == "imag":
        return np.zeros_like(phi_real)
    raise ValueError(f"Unknown mode={mode!r}. Use 'real' or 'abs'.")


def get_batch_for_id(dataset, id_):
    """Возвращает батч (coords, shape, patch) для всех точек одного ID."""
    slc = dataset.get_id_slice(id_)
    if slc.stop - slc.start == 0:
        return None
    items = [dataset[i] for i in range(slc.start, slc.stop)]
    coords = torch.cat([item["coords"].unsqueeze(0) for item in items], dim=0).to(device)
    shape = torch.cat([item["shape_params"].unsqueeze(0) for item in items], dim=0).to(device)
    patch = torch.cat([item["coarse_patch"].unsqueeze(0) for item in items], dim=0).to(device)
    return coords, shape, patch


def predict_fields(model, coords, shape, patch, fields_mean, fields_std):
    """Денормализует предсказания модели."""
    with torch.no_grad():
        pred = model(coords, shape, patch).detach().cpu().numpy()
    return pred * fields_std + fields_mean


def get_true_voltage_from_dataset(dataset, id_):
    row = dataset.df[dataset.df["id"] == id_].iloc[0]
    return row["voltage_complex"]


model = SRPINN(
    n_spatial=3,
    n_shape_params=2,
    n_coarse_nodes=8,
    n_field_vars=7,
    hidden_dim=256,
    n_blocks=10,
)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

train_ids, val_ids, test_ids = parse_idx_train(PATH_TO_FILES)
print(f"Train IDs: {train_ids[:5]}... (total {len(train_ids)})")

fields_mean, fields_std = compute_stats_incremental(
    generate_fine_fields(train_ids, PATH_TO_FILES)
)
print(f"Fields stats from train fine data: mean[6]={fields_mean[6]:.3e}, std[6]={fields_std[6]:.3e}")

fine_df = load_all_csv_cached(PATH_TO_FILES, 'results_fine', 'fine_df')
fine_df["voltage_complex"] = fine_df["voltage"].apply(parse_complex)

all_ids = sorted(fine_df["id"].unique())
dataset = CylinderStressDataset(
    PATH_TO_FILES=PATH_TO_FILES,
    csv_df=fine_df,
    ids=all_ids,
    mesh_type="fine",
    n_neighbors=8,
    normalize=True,
    external_stats=(fields_mean, fields_std),
    subsample_ratio=1.0
)

coarse_coords_list = []
coarse_fields_list = []
coarse_ids_list = []
for id_ in all_ids:
    try:
        coords, fields = load_mat_with_cache(id_, "coarse", PATH_TO_FILES)
        coarse_coords_list.append(coords)
        coarse_fields_list.append(fields)
        coarse_ids_list.append(id_)
    except FileNotFoundError:
        print(f"Warning: coarse file for ID {id_} not found, skipping")

if coarse_coords_list:
    dataset.set_coarse_data(coarse_coords_list, coarse_fields_list, coarse_ids_list)

def evaluate_voltage(model, dataset, device):
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
            pred_fields = predict_fields(
                model, coords, shape, patch,
                dataset.fields_mean_np, dataset.fields_std_np
            )
            phi_pred = pred_fields[:, 6]

            id_idx = dataset.id_to_index[id_]
            bottom_mask = dataset.bottom_mask_list[id_idx]
            top_mask = dataset.top_mask_list[id_idx]
            if not np.any(bottom_mask) or not np.any(top_mask):
                continue

            phi_bottom = phi_pred[bottom_mask].mean()
            phi_top = phi_pred[top_mask].mean()
            V_pred = phi_top - phi_bottom
            V_true = get_true_voltage_from_dataset(dataset, id_)
            V_true_abs = abs(V_true)

            V_preds.append(V_pred)
            V_trues.append(V_true_abs)
            errors.append(abs(V_pred - V_true_abs) / (V_true_abs + EPS))

    return np.array(V_preds), np.array(V_trues), np.array(errors)


V_preds, V_trues, errors = evaluate_voltage(model, dataset, device)

print(f"Средняя относительная ошибка напряжения: {np.mean(errors):.4f}")
print(f"Медианная относительная ошибка: {np.median(errors):.4f}")

mask = np.isfinite(V_trues) & np.isfinite(V_preds)
V_trues_finite = V_trues[mask]
V_preds_finite = V_preds[mask]

reg = LinearRegression(fit_intercept=True)
reg.fit(V_trues_finite.reshape(-1, 1), V_preds_finite.reshape(-1, 1))
a = reg.coef_[0][0]
b = reg.intercept_[0]
r2 = reg.score(V_trues_finite.reshape(-1, 1), V_preds_finite.reshape(-1, 1))

x_line = np.linspace(V_trues_finite.min(), V_trues_finite.max(), 100)
y_line = a * x_line + b

plt.figure(figsize=(8, 6))
plt.scatter(V_trues_finite, V_preds_finite, alpha=0.7, label="Данные")
plt.plot(x_line, y_line, "g-", linewidth=2, label=f"МНК: V_pred = {a:.3f}·V_true + {b:.3e}")
plt.plot([V_trues_finite.min(), V_trues_finite.max()],
         [V_trues_finite.min(), V_trues_finite.max()], "r--", label="Идеал (y=x)")
plt.xlabel("Истинное напряжение (модуль)")
plt.ylabel("Предсказанное напряжение")
plt.title("Сравнение напряжения на торцах")
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, f"$R^2 = {r2:.4f}$", transform=plt.gca().transAxes,
         verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.savefig("voltage_scatter_with_lr.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(errors[mask], bins=30, edgecolor="black")
plt.xlabel("Относительная ошибка |V_pred - V_true|/|V_true|")
plt.ylabel("Количество ID")
plt.title("Распределение ошибок напряжения")
plt.grid(True)
plt.savefig("voltage_errors_hist.png", dpi=150, bbox_inches="tight")
plt.show()

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
    """Визуализация поля phi (реальная часть) на одном слое (максимальная амплитуда)."""
    fname = os.path.join(PATH_TO_FILES, f'pinndata_quick_id_{id_:04d}_fine.mat')
    with h5py.File(fname, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:]
        phi = load_complex_from_h5(f, 'phi')

    if vis_mode == "real":
        phi_true_all = np.real(phi)
        phi_label = "Re(phi)"
    elif vis_mode == "abs":
        phi_true_all = np.abs(phi)
        phi_label = "|phi|"
    else:
        phi_true_all = np.real(phi)
        phi_label = "Re(phi)"

    batch = get_batch_for_id(dataset, id_)
    if batch is None:
        print(f"ID {id_}: нет данных в датасете")
        return
    coords, shape, patch = batch
    pred_fields = predict_fields(
        model, coords, shape, patch,
        dataset.fields_mean_np, dataset.fields_std_np
    )
    phi_pred_flat = pred_fields[:, 6]

    phi_pred_all = phi_pred_flat.reshape(X.shape)
    z_means = np.mean(np.abs(phi_true_all), axis=(0, 1))
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

for id_ in all_ids[:3]:
    visualize_fields_improved(
        id_, model, device, dataset,
        vis_mode=VIS_FIELD_MODE,
        use_log_scale=USE_LOG_SCALE,
        save_fig=True
    )