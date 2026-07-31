import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import pandas as pd
import glob
import re
from scipy.spatial import cKDTree

# Импорты из модулей проекта
from src.data_utils import (
    load_mat_with_cache,
    load_all_csv,
    parse_complex,
    build_kdtree,
    compute_patch_for_points
)
from src.model import SRPINN
from src.config import PATH_TO_FILES, CACHE_DIR  # если нужно, или задаём явно

# ------------------- Вспомогательные функции -------------------
def get_test_ids(data_dir: str) -> list:
    """Возвращает список ID, для которых существуют fine-файлы."""
    pattern = os.path.join(data_dir, "pinndata_quick_id_*_fine.mat")
    files = glob.glob(pattern)
    ids = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'id_(\d+)_fine', basename)
        if match:
            ids.append(int(match.group(1)))
    return sorted(set(ids))

# ------------------- Основная функция визуализации -------------------
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
    model = SRPINN(n_field_vars=7)   # число компонент в полях (без мнимой части phi)
    state_dict = checkpoint['model_state_dict']
    # Убираем префикс _orig_mod., если он есть (после torch.compile)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len('_orig_mod.'):] if key.startswith('_orig_mod.') else key
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

    # 4. Параметры формы из CSV (fine или coarse)
    df = load_all_csv(data_dir, 'results_fine')  # использует кэш
    if df.empty:
        df = load_all_csv(data_dir, 'results_coarse')
    matching_rows = df[df['id'] == target_id]
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

    # 8. Предсказание модели (батчами)
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

    eps_rel = 1e-20
    ux_rel_err_model = np.abs(ux_pred - ux_true) / (np.abs(ux_true) + eps_rel)
    uy_rel_err_model = np.abs(uy_pred - uy_true) / (np.abs(uy_true) + eps_rel)

    # 9. Триангуляция для визуализации
    x = top_coords[:, 0]
    y = top_coords[:, 1]
    tri = Triangulation(x, y)

    # ---- Первая фигура: модель vs fine (с клиппингом ошибки) ----
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
    # ux: относительная ошибка
    ax3 = fig1.add_subplot(2, 3, 3)
    tcf3 = ax3.tripcolor(tri, ux_rel_err_model, shading='gouraud', cmap='hot')
    ax3.set_title('Relative error (model)')
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
    # uy: относительная ошибка
    ax6 = fig1.add_subplot(2, 3, 6)
    tcf6 = ax6.tripcolor(tri, uy_rel_err_model, shading='gouraud', cmap='hot')
    ax6.set_title('Relative error (model)')
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

    # ---- Вторая фигура: coarse vs fine (без клиппинга) ----
    # Найдём точки верхней грани в coarse-сетке
    z_coarse = coarse_coords[:, 2]
    z_min_c, z_max_c = z_coarse.min(), z_coarse.max()
    eps_z_c = 1e-4 * (z_max_c - z_min_c)
    top_mask_coarse = np.abs(z_coarse - z_max_c) < eps_z_c
    coarse_top_coords = coarse_coords[top_mask_coarse]
    coarse_top_fields = coarse_fields[top_mask_coarse]

    # Интерполяция coarse полей на точки fine (по ближайшему соседу в плоскости xy)
    tree_coarse_xy = cKDTree(coarse_top_coords[:, :2])
    _, idxs = tree_coarse_xy.query(top_coords[:, :2], k=1)
    coarse_fields_at_fine = coarse_top_fields[idxs]
    ux_coarse_interp = coarse_fields_at_fine[:, 0]
    uy_coarse_interp = coarse_fields_at_fine[:, 2]

    ux_rel_err_coarse = np.abs(ux_coarse_interp - ux_true) / (np.abs(ux_true) + eps_rel)
    uy_rel_err_coarse = np.abs(uy_coarse_interp - uy_true) / (np.abs(uy_true) + eps_rel)

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
    # ux: относительная ошибка
    im3 = axes2[0,2].tripcolor(tri, ux_rel_err_coarse, shading='gouraud', cmap='hot')
    axes2[0,2].set_title('Relative error |coarse - fine|')
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
    # uy: относительная ошибка
    im6 = axes2[1,2].tripcolor(tri, uy_rel_err_coarse, shading='gouraud', cmap='hot')
    axes2[1,2].set_title('Relative error |coarse - fine|')
    axes2[1,2].axis('equal'); axes2[1,2].set_xlabel('x (m)'); axes2[1,2].set_ylabel('y (m)')
    plt.colorbar(im6, ax=axes2[1,2], label='relative error')
    plt.tight_layout()
    if save_fig:
        out_path2 = os.path.join(output_dir, f'coarse_vs_fine_id_{target_id}.png')
        plt.savefig(out_path2, dpi=200)
        print(f"График coarse vs fine сохранён в {out_path2}")
    else:
        plt.show()

    # ---- Вывод численных метрик и напряжения ----
    mean_rel_err_coarse_ux = np.mean(ux_rel_err_coarse)
    mean_rel_err_coarse_uy = np.mean(uy_rel_err_coarse)
    mean_rel_err_model_ux = np.mean(ux_rel_err_model)
    mean_rel_err_model_uy = np.mean(uy_rel_err_model)
    print(f"Средняя относительная ошибка (coarse -> fine): ux = {mean_rel_err_coarse_ux:.4f}, uy = {mean_rel_err_coarse_uy:.4f}")
    print(f"Средняя относительная ошибка (модель -> fine): ux = {mean_rel_err_model_ux:.4f}, uy = {mean_rel_err_model_uy:.4f}")

    # Вычисление напряжения (phi_top - phi_bottom)
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

# ------------------- Точка входа -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация ux и uy на верхнем торце с сравнением")
    parser.add_argument("--id", type=int, default=None, help="ID цилиндра")
    parser.add_argument("--save", action="store_true", help="Сохранить график в файл")
    parser.add_argument("--model", type=str, default="best_srpinn_model_with_stats.pth", help="Путь к модели")
    parser.add_argument("--data_dir", type=str, default=PATH_TO_FILES, help="Папка с данными")
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
