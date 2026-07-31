import os
import glob
import re
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Iterable, Tuple, List, Optional, Dict
import torch
import logging

from config import CACHE_DIR

_MAT_CACHE = {}


def load_mat_with_cache(id_: int, mesh_type: str, data_dir: str) -> tuple:
    """
    Возвращает (coords, fields) для указанного ID и типа сетки.
    Использует кэширование в папке CACHE_DIR.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'id_{id_:04d}_{mesh_type}.npz')

    key = (id_, mesh_type)
    if key in _MAT_CACHE:
        return _MAT_CACHE[key]

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        coords = data['coords']
        fields = data['fields']
        return coords, fields

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
    _MAT_CACHE[key] = (coords, fields)
    return coords, fields


def load_all_csv_cached(data_dir: str, pattern: str, cache_name: str) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f'{cache_name}.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    csv_files = glob.glob(os.path.join(data_dir, f"{pattern}*.csv"))
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    df.to_parquet(cache_file)
    return df


def load_all_csv(data_dir: str, pattern: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, f"{pattern}*.csv"))
    if not csv_files:
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)


def compute_stats_incremental(
        arrays: Iterable[np.ndarray],
        epsilon: float = 1e-20,
        dtype: np.dtype = np.float64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет среднее и стандартное отклонение (population) по наборам строк,
    объединяя все массивы по вертикали, но без физического копирования данных.
    """
    n_total = 0
    sum_ = None
    sum_sq = None

    for arr in arrays:
        if arr.shape[0] == 0:
            continue
        if sum_ is None:
            d = arr.shape[1]
            sum_ = np.zeros(d, dtype=dtype)
            sum_sq = np.zeros(d, dtype=dtype)

        n_total += arr.shape[0]
        sum_ += arr.sum(axis=0, dtype=dtype)
        sum_sq += (arr.astype(dtype) ** 2).sum(axis=0)

    if n_total == 0:
        if sum_ is None:
            dim = 7
            mean = np.zeros(dim)
            std = np.ones(dim) * epsilon
        else:
            mean = np.zeros_like(sum_)
            std = np.ones_like(sum_) * epsilon
        return mean, std

    mean = sum_ / n_total
    variance = sum_sq / n_total - mean * mean
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, epsilon)
    return mean, std


def generate_fine_fields(train_ids, data_dir):
    """Генератор, выдающий fields для каждого существующего fine-файла."""
    for id_ in train_ids:
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_fine.mat')
        if not os.path.exists(fname):
            continue
        _, fields = load_mat_with_cache(id_, "fine", data_dir)
        yield fields


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


def build_kdtree(coords: np.ndarray) -> cKDTree:
    return cKDTree(coords)

def to_device(batch, device):
    """Переносит все тензоры в словаре на указанное устройство."""
    if batch is None:
        return None
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


def find_coarse_patch(coarse_tree: cKDTree, coarse_coords: np.ndarray,
                      coarse_fields: np.ndarray, query_point: np.ndarray, n_neighbors: int = 8) -> np.ndarray:
    dists, idxs = coarse_tree.query(query_point, k=n_neighbors)
    neighbor_coords = coarse_coords[idxs]
    neighbor_fields = coarse_fields[idxs]
    scale = np.mean(dists) + 1e-8
    rel_coords = (neighbor_coords - query_point) / scale
    patch = np.concatenate([neighbor_fields.ravel(), rel_coords.ravel()])
    return patch


def prepare_datasets(data_dir: str,
                     coarse_ids: List[int],
                     train_ids: List[int],
                     val_ids: List[int],
                     n_neighbors: int = 8):
    """
    Подготовка всех датасетов с правильной нормализацией:
    - Статистика mean/std для полей считается ТОЛЬКО по fine-данным (target)
    - Coarse-данные используются только для patch (не для статистики)
    """
    
    from datasets import CylinderStressDataset, CollocationDataset

    fine_df = load_all_csv_cached(data_dir, 'results_fine', 'fine_df')
    coarse_df = load_all_csv_cached(data_dir, 'results_coarse', 'coarse_df')

    if fine_df.empty:
        raise ValueError("Не найдены results_fine*.csv")

    fields_mean, fields_std = compute_stats_incremental(generate_fine_fields(train_ids, data_dir))

    print(f"Fields stats FROM FINE data (total {len(fields_mean)} components):")
    for i in range(len(fields_mean)):
        print(f"  mean[{i}]={fields_mean[i]:.3e}, std[{i}]={fields_std[i]:.3e}")

    external_stats = (fields_mean, fields_std)

    train_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=train_ids,
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True,
        external_stats=external_stats,
        subsample_ratio=1.0
    )

    val_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=val_ids,
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True,
        external_stats=external_stats,
        subsample_ratio=1.0
    )

    coarse_coords_list = []
    coarse_fields_list = []
    coarse_id_list = []
    needed_ids = set(train_ids + val_ids)

    for id_ in needed_ids:
        fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_coarse.mat')
        if not os.path.exists(fname):
            print(f"Warning: coarse file for ID {id_} not found")
            continue
        coords, fields = load_mat_with_cache(id_, "coarse", data_dir)
        coarse_coords_list.append(coords)
        coarse_fields_list.append(fields)
        coarse_id_list.append(id_)

    def set_coarse(ds, ids):
        idxs = [i for i, cid in enumerate(coarse_id_list) if cid in ids]
        if idxs:
            ds.set_coarse_data(
                [coarse_coords_list[i] for i in idxs],
                [coarse_fields_list[i] for i in idxs],
                [coarse_id_list[i] for i in idxs]
            )

    set_coarse(train_dataset, train_ids)
    set_coarse(val_dataset, val_ids)

    coarse_shape_dict = {
        row['id']: (row['r_um'], row['h_um'])
        for _, row in coarse_df.iterrows()
    }
    coarse_id_to_index = {id_: idx for idx, id_ in enumerate(coarse_id_list)}

    coarse_trees_cache = {}
    for idx, id_ in enumerate(coarse_id_list):
        coarse_trees_cache[id_] = build_kdtree(coarse_coords_list[idx])

    colloc_ids = [id_ for id_ in train_ids if id_ in needed_ids]
    coarse_data_colloc = {}
    shape_params_colloc = {}

    for id_ in colloc_ids:
        idx = coarse_id_to_index.get(id_)
        if idx is None:
            continue
        coarse_data_colloc[id_] = (
            coarse_trees_cache[id_],
            coarse_coords_list[idx],
            coarse_fields_list[idx]
        )
        if id_ in coarse_shape_dict:
            shape_params_colloc[id_] = coarse_shape_dict[id_]

    colloc_dataset = CollocationDataset(
        ids=colloc_ids,
        shape_params=shape_params_colloc,
        coarse_data=coarse_data_colloc,
        n_points_per_id=200,
        n_neighbors=n_neighbors,
        normalize=True,
        coords_stats=(train_dataset.coords_mean, train_dataset.coords_std),
        shape_stats=(train_dataset.shape_mean, train_dataset.shape_std),
        fields_stats=external_stats
    )

    stats = {
        'coords_mean': train_dataset.coords_mean,
        'coords_std': train_dataset.coords_std,
        'shape_mean': train_dataset.shape_mean,
        'shape_std': train_dataset.shape_std,
        'fields_mean': torch.tensor(fields_mean, dtype=torch.float32),
        'fields_std': torch.tensor(fields_std, dtype=torch.float32)
    }

    return train_dataset, val_dataset, colloc_dataset, stats

def compute_patch_for_points(
    points: np.ndarray,
    coarse_tree: cKDTree,
    coarse_coords: np.ndarray,
    coarse_fields: np.ndarray,
    n_neighbors: int = 8,
    fields_mean: Optional[np.ndarray] = None,
    fields_std: Optional[np.ndarray] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Вычисляет патчи для заданных точек на основе coarse-данных.
    Возвращает массив shape (N, n_neighbors * (n_field_vars + 3)).
    """
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

def load_coarse_voltage_data(data_dir="./files/"):
    """Загружает все results_coarse*.csv, возвращает DataFrame с r_um, h_um, |V|."""
    csv_files = glob.glob(data_dir + "results_coarse*.csv")
    if not csv_files:
        raise FileNotFoundError("Не найдены results_coarse*.csv в папке ./files/")
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['voltage_complex'] = df['voltage'].apply(parse_complex)
        df['V_abs'] = df['voltage_complex'].apply(abs)
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.dropna(subset=['r_um', 'h_um', 'V_abs'])
    return df_all[['id', 'r_um', 'h_um', 'V_abs']]
