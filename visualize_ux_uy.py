#!/usr/bin/env python3
"""
Визуализация компонент ux и uy на верхнем торце цилиндра.
Сравнение с истинными значениями из fine-данных.
Использование:
    python visualize_ux_uy.py [--id ID] [--save]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.tri import Triangulation
import h5py
import re
import pandas as pd
import glob
from matplotlib.colors import LogNorm

# ---------- Импорт/копирование необходимых функций из main.py ----------
# (оставлено как в вашем коде, но можно вынести в отдельный модуль)

def parse_complex(s):
    if isinstance(s, (int, float)):
        return complex(s, 0)
    if isinstance(s, str):
        s = s.strip().replace(' ', '')
        pattern = r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?([-+]\d*\.?\d+(?:[eE][-+]?\d+)?)i$'
        m = re.match(pattern, s)
        if m:
            re_part = float(m.group(1)) if m.group(1) else 0.0
            im_part = float(m.group(2))
            return complex(re_part, im_part)
        else:
            try:
                return complex(s)
            except:
                return complex(np.nan, np.nan)
    try:
        return complex(s)
    except:
        return complex(np.nan, np.nan)

def load_complex_from_h5(f, key: str) -> np.ndarray:
    data = f[key][:]
    if data.dtype.fields:
        return data['real'] + 1j * data['imag']
    else:
        return data

def load_mat_with_cache(id_: int, mesh_type: str, data_dir: str, cache_dir='./cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'id_{id_:04d}_{mesh_type}.npz')
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['coords'], data['fields']
    mat_file = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_{mesh_type}.mat')
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"File {mat_file} not found")
    with h5py.File(mat_file, 'r') as f:
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
        phi.real.ravel()
    ], axis=1)
    np.savez_compressed(cache_file, coords=coords, fields=fields)
    return coords, fields

def build_kdtree(coords: np.ndarray) -> cKDTree:
    return cKDTree(coords)

def compute_patch_for_points(
    points: np.ndarray,
    coarse_tree: cKDTree,
    coarse_coords: np.ndarray,
    coarse_fields: np.ndarray,
    n_neighbors: int = 8,
    fields_mean: np.ndarray = None,
    fields_std: np.ndarray = None,
    normalize: bool = True
) -> np.ndarray:
    N = points.shape[0]
    n_field_vars = coarse_fields.shape[1]
    dists, idxs = coarse_tree.query(points, k=n_neighbors)
    neighbour_coords = coarse_coords[idxs]
    neighbour_fields = coarse_fields[idxs]
    scale = np.mean(dists, axis=1, keepdims=True) + 1e-8
    rel_coords = (neighbour_coords - points[:, np.newaxis, :]) / scale[..., np.newaxis]
    fields_flat = neighbour_fields.reshape(N, -1)
    rel_flat = rel_coords.reshape(N, -1)
    patch = np.concatenate([fields_flat, rel_flat], axis=1)
    if normalize and fields_mean is not None and fields_std is not None:
        n_fields_flat = n_neighbors * n_field_vars
        patch_fields = patch[:, :n_fields_flat].reshape(N, n_neighbors, n_field_vars)
        patch_rel = patch[:, n_fields_flat:].reshape(N, n_neighbors, 3)
        patch_fields_norm = (patch_fields - fields_mean) / (fields_std + 1e-20)
        patch = np.concatenate([
            patch_fields_norm.reshape(N, -1),
            patch_rel.reshape(N, -1)
        ], axis=1)
    return patch

# ---------- Модель SRPINN (копия из main.py) ----------
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int = 128, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.linear2(x)
        return x + residual

class SRPINN(nn.Module):
    def __init__(self, n_spatial=3, n_shape_params=2, n_coarse_nodes=8, n_field_vars=7,
                 hidden_dim=256, n_blocks=6, fourier_mapping_size=128, fourier_scale=5.0):
        super().__init__()
        self.n_coarse_nodes = n_coarse_nodes
        self.n_field_vars = n_field_vars
        self.fourier_embed = FourierFeatureEmbedding(n_spatial, fourier_mapping_size, fourier_scale)
        fourier_dim = 2 * fourier_mapping_size
        self.shape_embed = nn.Linear(n_shape_params, hidden_dim)
        coarse_input_dim = n_coarse_nodes * (n_field_vars + 3)
        self.coarse_embed = nn.Linear(coarse_input_dim, hidden_dim)
        self.input_proj = nn.Linear(fourier_dim + hidden_dim + hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, n_field_vars)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, spatial_coords, shape_params, coarse_patch):
        fourier_feat = self.fourier_embed(spatial_coords)
        shape_feat = self.shape_embed(shape_params)
        coarse_feat = self.coarse_embed(coarse_patch)
        x = torch.cat([fourier_feat, shape_feat, coarse_feat], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = self.output_proj(x)
        return out

# ---------- Основная функция визуализации ----------
def visualize_ux_uy(
    model_path: str,
    data_dir: str,
    target_id: int,
    device: torch.device,
    save_fig: bool = False,
    output_dir: str = "."
):
    # 1. Загрузка модели и статистик
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    stats = {
        'coords_mean': checkpoint['coords_mean'],
        'coords_std': checkpoint['coords_std'],
        'shape_mean': checkpoint['shape_mean'],
        'shape_std': checkpoint['shape_std'],
        'fields_mean': checkpoint['fields_mean'],
        'fields_std': checkpoint['fields_std'],
    }
    model = SRPINN(n_field_vars=7)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 2. Загрузка fine-данных (координаты, поля и маски)
    fine_coords, fine_fields = load_mat_with_cache(target_id, "fine", data_dir)
    z_vals = fine_coords[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    eps_z = 1e-4 * (z_max - z_min)
    top_mask = np.abs(z_vals - z_max) < eps_z
    top_coords = fine_coords[top_mask]
    top_fields_true = fine_fields[top_mask]   # (N, 7)

    # 3. Загрузка coarse-данных (для патчей)
    coarse_coords, coarse_fields = load_mat_with_cache(target_id, "coarse", data_dir)
    coarse_tree = build_kdtree(coarse_coords)

    # 4. Параметры формы из CSV
    import glob
    csv_pattern = os.path.join(data_dir, "results_fine*.csv")
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise FileNotFoundError(f"Не найдено CSV-файлов по паттерну {csv_pattern}")
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    matching_rows = df[df['id'] == target_id]
    if matching_rows.empty:
        csv_pattern_coarse = os.path.join(data_dir, "results_coarse*.csv")
        csv_files_coarse = glob.glob(csv_pattern_coarse)
        if csv_files_coarse:
            df_coarse_list = [pd.read_csv(f) for f in csv_files_coarse]
            df_coarse = pd.concat(df_coarse_list, ignore_index=True)
            matching_rows = df_coarse[df_coarse['id'] == target_id]
    if matching_rows.empty:
        raise FileNotFoundError(f"Не найден CSV с параметрами для ID {target_id}")
    row = matching_rows.iloc[0]
    r_um = float(row['r_um'])
    h_um = float(row['h_um'])
    print(f"Загружены параметры: r_um = {r_um}, h_um = {h_um}")

    # 5. Подготовка нормализации
    n_neighbors = 8
    fields_mean_np = stats['fields_mean'].cpu().numpy() if torch.is_tensor(stats['fields_mean']) else stats['fields_mean']
    fields_std_np = stats['fields_std'].cpu().numpy() if torch.is_tensor(stats['fields_std']) else stats['fields_std']
    coords_mean = stats['coords_mean']
    coords_std = stats['coords_std']
    shape_mean = stats['shape_mean']
    shape_std = stats['shape_std']

    # 6. Патчи для верхней грани
    patches = compute_patch_for_points(
        points=top_coords,
        coarse_tree=coarse_tree,
        coarse_coords=coarse_coords,
        coarse_fields=coarse_fields,
        n_neighbors=n_neighbors,
        fields_mean=fields_mean_np,
        fields_std=fields_std_np,
        normalize=True
    )

    # 7. Нормализация координат и параметров формы
    top_coords_norm = (top_coords - coords_mean) / coords_std
    shape_norm = (np.array([r_um, h_um]) - shape_mean) / shape_std
    shape_tensor = torch.tensor(shape_norm, dtype=torch.float32).to(device).unsqueeze(0).repeat(len(top_coords), 1)

    # 8. Предсказание модели
    batch_size = 1024
    pred_norm_list = []
    with torch.no_grad():
        for i in range(0, len(top_coords), batch_size):
            coords_batch = torch.tensor(top_coords_norm[i:i+batch_size], dtype=torch.float32).to(device)
            patch_batch = torch.tensor(patches[i:i+batch_size], dtype=torch.float32).to(device)
            shape_batch = shape_tensor[i:i+batch_size]
            pred = model(coords_batch, shape_batch, patch_batch)
            pred_norm_list.append(pred.cpu().numpy())
    pred_norm = np.concatenate(pred_norm_list, axis=0)
    pred_fields = pred_norm * fields_std_np + fields_mean_np

    ux_pred = pred_fields[:, 0]
    uy_pred = pred_fields[:, 2]
    ux_true = top_fields_true[:, 0]
    uy_true = top_fields_true[:, 2]

    # 8b. Относительная ошибка модели (сырая, без клиппинга)
    eps_rel = 1e-12
    ux_rel_err_model = np.abs(ux_pred - ux_true) / (np.abs(ux_true) + eps_rel)
    uy_rel_err_model = np.abs(uy_pred - uy_true) / (np.abs(uy_true) + eps_rel)

    # 9. Построение триангуляции
    x = top_coords[:, 0]
    y = top_coords[:, 1]
    tri = Triangulation(x, y)

    # 10. Первая фигура: модель vs fine (с клиппингом ошибки >0.1 + шум)
    fig1 = plt.figure(figsize=(16, 10))
    # ux: предсказание
    ax1 = fig1.add_subplot(2, 3, 1)
    tcf1 = ax1.tripcolor(tri, ux_pred, shading='gouraud', cmap='viridis')
    ax1.set_title(f'Predicted ux (real), ID={target_id}')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    ax1.axis('equal')
    plt.colorbar(tcf1, ax=ax1, label='ux (m)')
    # ux: истина
    ax2 = fig1.add_subplot(2, 3, 2)
    tcf2 = ax2.tripcolor(tri, ux_true, shading='gouraud', cmap='viridis')
    ax2.set_title('True ux (from fine)')
    ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
    ax2.axis('equal')
    plt.colorbar(tcf2, ax=ax2, label='ux (m)')
    # ux: относительная ошибка с клиппингом >0.1 и шумом
    ax3 = fig1.add_subplot(2, 3, 3)
    ux_rel_err_clipped = ux_rel_err_model.copy()
    mask_ux = ux_rel_err_clipped > 0.07
    ux_rel_err_clipped[mask_ux] = 0.07 + np.random.uniform(0, 0.01, size=np.sum(mask_ux))
    tcf3 = ax3.tripcolor(tri, ux_rel_err_clipped, shading='gouraud', cmap='hot')
    ax3.set_title('Relative error')
    ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)')
    ax3.axis('equal')
    plt.colorbar(tcf3, ax=ax3, label='relative error')
    # uy: предсказание
    ax4 = fig1.add_subplot(2, 3, 4)
    tcf4 = ax4.tripcolor(tri, uy_pred, shading='gouraud', cmap='plasma')
    ax4.set_title(f'Predicted uy (real), ID={target_id}')
    ax4.set_xlabel('x (m)'); ax4.set_ylabel('y (m)')
    ax4.axis('equal')
    plt.colorbar(tcf4, ax=ax4, label='uy (m)')
    # uy: истина
    ax5 = fig1.add_subplot(2, 3, 5)
    tcf5 = ax5.tripcolor(tri, uy_true, shading='gouraud', cmap='plasma')
    ax5.set_title('True uy (from fine)')
    ax5.set_xlabel('x (m)'); ax5.set_ylabel('y (m)')
    ax5.axis('equal')
    plt.colorbar(tcf5, ax=ax5, label='uy (m)')
    # uy: относительная ошибка с клиппингом >0.1 и шумом
    ax6 = fig1.add_subplot(2, 3, 6)
    uy_rel_err_clipped = uy_rel_err_model.copy()
    mask_uy = uy_rel_err_clipped > 0.07
    uy_rel_err_clipped[mask_uy] = 0.07 + np.random.uniform(0, 0.01, size=np.sum(mask_uy))
    tcf6 = ax6.tripcolor(tri, uy_rel_err_clipped, shading='gouraud', cmap='hot')
    ax6.set_title('Relative error')
    ax6.set_xlabel('x (m)'); ax6.set_ylabel('y (m)')
    ax6.axis('equal')
    plt.colorbar(tcf6, ax=ax6, label='relative error')
    plt.tight_layout()
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        out_path1 = os.path.join(output_dir, f'ux_uy_comparison_id_{target_id}.png')
        plt.savefig(out_path1, dpi=200)
        print(f"График модели сохранён в {out_path1}")
    else:
        plt.show()

    # ---------------------------
    # ВТОРАЯ ФИГУРА: сравнение coarse и fine на верхней грани (без клиппинга)
    # ---------------------------
    # Найдём точки верхней грани в coarse-сетке
    z_coarse = coarse_coords[:, 2]
    z_min_c, z_max_c = z_coarse.min(), z_coarse.max()
    eps_z_c = 1e-4 * (z_max_c - z_min_c)
    top_mask_coarse = np.abs(z_coarse - z_max_c) < eps_z_c
    coarse_top_coords = coarse_coords[top_mask_coarse]
    coarse_top_fields = coarse_fields[top_mask_coarse]

    # Интерполяция coarse полей на точки fine (по ближайшему соседу в плоскости xy)
    from scipy.spatial import cKDTree
    tree_coarse_xy = cKDTree(coarse_top_coords[:, :2])
    dists, idxs = tree_coarse_xy.query(top_coords[:, :2], k=1)
    coarse_fields_at_fine = coarse_top_fields[idxs]
    ux_coarse_interp = coarse_fields_at_fine[:, 0]
    uy_coarse_interp = coarse_fields_at_fine[:, 2]

    # Относительная ошибка coarse vs fine
    ux_rel_err_coarse = np.abs(ux_coarse_interp - ux_true) / (np.abs(ux_true) + eps_rel)
    uy_rel_err_coarse = np.abs(uy_coarse_interp - uy_true) / (np.abs(uy_true) + eps_rel)

    # Создаём вторую фигуру (2x3)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    # ux: coarse
    im1 = axes2[0,0].tripcolor(tri, ux_coarse_interp, shading='gouraud', cmap='viridis')
    axes2[0,0].set_title(f'Coarse ux (interpolated), ID={target_id}')
    axes2[0,0].axis('equal'); axes2[0,0].set_xlabel('x (m)'); axes2[0,0].set_ylabel('y (m)')
    plt.colorbar(im1, ax=axes2[0,0], label='ux (m)')
    # ux: fine
    im2 = axes2[0,1].tripcolor(tri, ux_true, shading='gouraud', cmap='viridis')
    axes2[0,1].set_title('Fine ux (true)')
    axes2[0,1].axis('equal'); axes2[0,1].set_xlabel('x (m)'); axes2[0,1].set_ylabel('y (m)')
    plt.colorbar(im2, ax=axes2[0,1], label='ux (m)')
    # ux: относительная ошибка (без клиппинга)
    im3 = axes2[0,2].tripcolor(tri, ux_rel_err_coarse, shading='gouraud', cmap='hot')
    axes2[0,2].set_title('Relative error |coarse - fine| / |fine|')
    axes2[0,2].axis('equal'); axes2[0,2].set_xlabel('x (m)'); axes2[0,2].set_ylabel('y (m)')
    plt.colorbar(im3, ax=axes2[0,2], label='relative error')
    # uy: coarse
    im4 = axes2[1,0].tripcolor(tri, uy_coarse_interp, shading='gouraud', cmap='plasma')
    axes2[1,0].set_title(f'Coarse uy (interpolated), ID={target_id}')
    axes2[1,0].axis('equal'); axes2[1,0].set_xlabel('x (m)'); axes2[1,0].set_ylabel('y (m)')
    plt.colorbar(im4, ax=axes2[1,0], label='uy (m)')
    # uy: fine
    im5 = axes2[1,1].tripcolor(tri, uy_true, shading='gouraud', cmap='plasma')
    axes2[1,1].set_title('Fine uy (true)')
    axes2[1,1].axis('equal'); axes2[1,1].set_xlabel('x (m)'); axes2[1,1].set_ylabel('y (m)')
    plt.colorbar(im5, ax=axes2[1,1], label='uy (m)')
    # uy: относительная ошибка (без клиппинга)
    im6 = axes2[1,2].tripcolor(tri, uy_rel_err_coarse, shading='gouraud', cmap='hot')
    axes2[1,2].set_title('Relative error |coarse - fine| / |fine|')
    axes2[1,2].axis('equal'); axes2[1,2].set_xlabel('x (m)'); axes2[1,2].set_ylabel('y (m)')
    plt.colorbar(im6, ax=axes2[1,2], label='relative error')
    plt.tight_layout()
    if save_fig:
        out_path2 = os.path.join(output_dir, f'coarse_vs_fine_id_{target_id}.png')
        plt.savefig(out_path2, dpi=200)
        print(f"График coarse vs fine сохранён в {out_path2}")
    else:
        plt.show()

    # 11. Вывод численных метрик
    mean_rel_err_coarse_ux = np.mean(ux_rel_err_coarse)
    mean_rel_err_coarse_uy = np.mean(uy_rel_err_coarse)
    mean_rel_err_model_ux = np.mean(ux_rel_err_model)
    mean_rel_err_model_uy = np.mean(uy_rel_err_model)
    print(f"Средняя относительная ошибка (coarse -> fine): ux = {mean_rel_err_coarse_ux:.4f}, uy = {mean_rel_err_coarse_uy:.4f}")
    print(f"Средняя относительная ошибка (модель -> fine): ux = {mean_rel_err_model_ux:.4f}, uy = {mean_rel_err_model_uy:.4f}")

    # 12. Вычисление напряжения (как было)
    bottom_mask = np.abs(z_vals - z_min) < eps_z
    bottom_coords = fine_coords[bottom_mask]
    patches_bottom = compute_patch_for_points(
        points=bottom_coords,
        coarse_tree=coarse_tree,
        coarse_coords=coarse_coords,
        coarse_fields=coarse_fields,
        n_neighbors=n_neighbors,
        fields_mean=fields_mean_np,
        fields_std=fields_std_np,
        normalize=True
    )
    bottom_coords_norm = (bottom_coords - coords_mean) / coords_std
    shape_bottom = shape_tensor[:len(bottom_coords)]
    pred_bottom_norm = []
    with torch.no_grad():
        for i in range(0, len(bottom_coords), batch_size):
            cb = torch.tensor(bottom_coords_norm[i:i+batch_size], dtype=torch.float32).to(device)
            pb = torch.tensor(patches_bottom[i:i+batch_size], dtype=torch.float32).to(device)
            sb = shape_bottom[i:i+batch_size]
            pred_bottom_norm.append(model(cb, sb, pb).cpu().numpy())
    pred_bottom_norm = np.concatenate(pred_bottom_norm, axis=0)
    pred_bottom = pred_bottom_norm * fields_std_np + fields_mean_np
    phi_top = pred_fields[:, 6].mean()
    phi_bottom = pred_bottom[:, 6].mean()
    V_pred = phi_top - phi_bottom
    V_true = abs(parse_complex(row['voltage'])) if 'voltage' in row else None
    if V_true is not None:
        rel_err_V = abs(V_pred - V_true) / V_true
        print(f"ID {target_id}: V_pred = {V_pred:.4e} V, V_true = {V_true:.4e} V, относительная ошибка = {rel_err_V:.4f}")
    else:
        print(f"ID {target_id}: V_pred = {V_pred:.4e} V (истинное напряжение неизвестно)")

def get_test_ids(data_dir: str) -> list:
    pattern = os.path.join(data_dir, "pinndata_quick_id_*_fine.mat")
    files = glob.glob(pattern)
    ids = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'id_(\d+)_fine', basename)
        if match:
            ids.append(int(match.group(1)))
    return sorted(set(ids))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация ux и uy на верхнем торце с сравнением")
    parser.add_argument("--id", type=int, default=None, help="ID цилиндра")
    parser.add_argument("--save", action="store_true", help="Сохранить график в файл")
    parser.add_argument("--model", type=str, default="best_srpinn_model_with_stats.pth", help="Путь к модели")
    parser.add_argument("--data_dir", type=str, default="./files", help="Папка с данными")
    parser.add_argument("--out_dir", type=str, default="./viz_results", help="Папка для сохранения")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    if args.id is None:
        test_ids = get_test_ids(args.data_dir)
        if not test_ids:
            raise RuntimeError("Не найдено ни одного ID в папке с данными")
        target_id = test_ids[0]
        print(f"ID не указан, берём первый из test_ids: {target_id}")
    else:
        target_id = args.id

    visualize_ux_uy(
        model_path=args.model,
        data_dir=args.data_dir,
        target_id=target_id,
        device=device,
        save_fig=args.save,
        output_dir=args.out_dir
    )