import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader
import os
import glob
import re
import h5py
from typing import Optional, Tuple, List, Dict, Iterable


path_to_files = "./files/"
CACHE_DIR = './cache'
torch.set_float32_matmul_precision('high')
torch._dynamo.config.disable = True

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

    # Кэша нет — читаем .mat
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
        phi.real.ravel(), phi.imag.ravel()
    ], axis=1)

    # Сохраняем в кэш
    np.savez_compressed(cache_file, coords=coords, fields=fields)
    _MAT_CACHE[key] = (coords, fields)
    return coords, fields


def load_all_csv_cached(data_dir: str, pattern: str, cache_name: str) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f'{cache_name}.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # Читаем все CSV
    csv_files = glob.glob(os.path.join(data_dir, f"{pattern}*.csv"))
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # Сохраняем в кэш
    df.to_parquet(cache_file)
    return df

def to_device(batch, device):
    """Переносит все тензоры в словаре на указанное устройство."""
    if batch is None:
        return None
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


def compute_stats_incremental(
    arrays: Iterable[np.ndarray],
    epsilon: float = 1e-20,
    dtype: np.dtype = np.float64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет среднее и стандартное отклонение (population) по наборам строк,
    объединяя все массивы по вертикали, но без физического копирования данных.

    Параметры
    ----------
    arrays : Iterable[np.ndarray]
        Итератор или список массивов. Каждый массив имеет форму (n_i, d).
        Размерность d определяется по первому непустому массиву.
    epsilon : float
        Нижняя граница для стандартного отклонения (чтобы избежать деления на ноль).
    dtype : np.dtype
        Тип данных для накопления сумм (по умолчанию float64).

    Возвращает
    ----------
    mean : np.ndarray
        Вектор средних значений для каждого из d столбцов.
    std : np.ndarray
        Вектор стандартных отклонений (pop) с обрезанием по epsilon.
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
            dim = 8
            mean = np.zeros(dim)
            std = np.ones(dim) * epsilon
        else:
            mean = np.zeros_like(sum_)
            std = np.ones_like(sum_) * epsilon
        return mean, std

    mean = sum_ / n_total
    variance = sum_sq / n_total - mean * mean
    variance = np.maximum(variance, 0.0)   # отсечь малые отрицательные из-за погрешностей
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
    # Handle numeric inputs directly
    if isinstance(s, (int, float)):
        return complex(s, 0)   # treat as real number
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
                return complex(s)   # e.g., "1.2" -> 1.2+0j
            except:
                return complex(np.nan, np.nan)
    # Fallback for any other type
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


def find_coarse_patch(coarse_tree: cKDTree, coarse_coords: np.ndarray,
                      coarse_fields: np.ndarray, query_point: np.ndarray, n_neighbors: int = 8) -> np.ndarray:
    dists, idxs = coarse_tree.query(query_point, k=n_neighbors)
    neighbor_coords = coarse_coords[idxs]
    neighbor_fields = coarse_fields[idxs]
    scale = np.mean(dists) + 1e-8
    rel_coords = (neighbor_coords - query_point) / scale
    patch = np.concatenate([neighbor_fields.ravel(), rel_coords.ravel()])
    return patch

def load_all_csv(data_dir: str, pattern: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, f"{pattern}*.csv"))
    if not csv_files:
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)


class CylinderStressDataset(Dataset):
    def __init__(self, data_dir: str, csv_df: pd.DataFrame, ids: List[int], mesh_type: str,
                 n_neighbors: int = 8, normalize: bool = True,
                 external_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 subsample_ratio: float = 1.0):
        self.data_dir = data_dir
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.mesh_type = mesh_type
        self.subsample_ratio = subsample_ratio
        self.df = csv_df[csv_df['id'].isin(ids)].reset_index(drop=True)
        self.df['voltage_complex'] = self.df['voltage'].apply(parse_complex)
        self.coords_list = []
        self.fields_list = []
        self.shape_params = []
        self.id_to_index = {}
        self.z_min_list = []
        self.z_max_list = []
        self.bottom_mask_list = []
        self.top_mask_list = []
        self.total_points = 0
        self.cumulative_sizes = [0]

        for idx, row in self.df.iterrows():
            id_ = int(row['id'])
            r_um = row['r_um']
            h_um = row['h_um']
            fname = os.path.join(data_dir, f'pinndata_quick_id_{id_:04d}_{mesh_type}.mat')
            if not os.path.exists(fname):
                print(f"Warning: file {fname} not found, skipping id {id_}")
                continue

            coords, fields = load_mat_with_cache(id_, mesh_type, self.data_dir)

            z_vals = coords[:, 2]
            z_min = z_vals.min()
            z_max = z_vals.max()
            self.z_min_list.append(z_min)
            self.z_max_list.append(z_max)
            eps_z = 1e-6 * (z_max - z_min)
            bottom_mask = np.abs(z_vals - z_min) < eps_z
            top_mask = np.abs(z_vals - z_max) < eps_z

            if subsample_ratio < 1.0:
                n_total = coords.shape[0]
                n_keep = int(n_total * subsample_ratio)
                idx_keep = np.random.choice(n_total, n_keep, replace=False)
                coords = coords[idx_keep]
                fields = fields[idx_keep]
                bottom_mask = bottom_mask[idx_keep]
                top_mask = top_mask[idx_keep]

            self.coords_list.append(coords)
            self.fields_list.append(fields)
            self.shape_params.append([r_um, h_um])
            self.id_to_index[id_] = len(self.coords_list) - 1
            self.bottom_mask_list.append(bottom_mask)
            self.top_mask_list.append(top_mask)
            self.total_points += coords.shape[0]
            self.cumulative_sizes.append(self.total_points)

        # Нормализация
        if normalize and self.total_points > 0:
            self.coords_mean, self.coords_std = compute_stats_incremental(self.coords_list, epsilon=1e-8)

            all_shape = np.array(self.shape_params)
            self.shape_mean = all_shape.mean(axis=0)
            self.shape_std = np.maximum(all_shape.std(axis=0), 1e-8)

            if external_stats is not None:
                fields_mean_np, fields_std_np = external_stats

                if torch.is_tensor(fields_mean_np):
                    fields_mean_np = fields_mean_np.cpu().numpy()
                if torch.is_tensor(fields_std_np):
                    fields_std_np = fields_std_np.cpu().numpy()
                self.fields_mean_np = fields_mean_np
                self.fields_std_np = fields_std_np
                print(f"Using external fields stats: mean[6]={self.fields_mean_np[6]:.3e}, std[6]={self.fields_std_np[6]:.3e}")
            else:
                self.fields_mean_np, self.fields_std_np = compute_stats_incremental(self.fields_list, epsilon=1e-20)
                print(f"Computed from fine: mean[6]={self.fields_mean_np[6]:.3e}, std[6]={self.fields_std_np[6]:.3e}")

            # Создаём тензоры для быстрого доступа в __getitem__
            self.fields_mean_tensor = torch.from_numpy(self.fields_mean_np).float()
            self.fields_std_tensor = torch.from_numpy(self.fields_std_np).float()
        else:
            self.coords_mean = np.zeros(3)
            self.coords_std = np.ones(3)
            self.shape_mean = np.zeros(2)
            self.shape_std = np.ones(2)
            self.fields_mean_np = np.zeros(8)
            self.fields_std_np = np.ones(8)
            self.fields_mean_tensor = torch.zeros(8, dtype=torch.float32)
            self.fields_std_tensor = torch.ones(8, dtype=torch.float32)

        self.coarse_trees = None
        self.coarse_coords = None
        self.coarse_fields = None
        self.precomputed_patches = None

    def set_coarse_data(self, coarse_coords_list, coarse_fields_list, coarse_ids):
        self.coarse_trees = {}
        self.coarse_coords = {}
        self.coarse_fields = {}
        for idx, id_ in enumerate(coarse_ids):
            if id_ in self.id_to_index:
                self.coarse_trees[id_] = build_kdtree(coarse_coords_list[idx])
                self.coarse_coords[id_] = coarse_coords_list[idx]
                self.coarse_fields[id_] = coarse_fields_list[idx]

        self._precompute_patches()

    def _precompute_patches(self):
        """Вычисляет и сохраняет патчи для всех точек всех ID (один раз) — ускоренная версия."""
        if self.coarse_trees is None:
            self.precomputed_patches = None
            return

        self.precomputed_patches = []
        for id_idx, coords in enumerate(self.coords_list):
            id_ = int(self.df.iloc[id_idx]['id'])
            if id_ not in self.coarse_trees:
                # Нет coarse-данных: создаём нулевые патчи
                patches = np.zeros((coords.shape[0], self.n_neighbors * (8 + 3)), dtype=np.float32)
                self.precomputed_patches.append(patches)
                continue

            tree = self.coarse_trees[id_]
            coarse_coords = self.coarse_coords[id_]
            coarse_fields = self.coarse_fields[id_]

            # 1. Batch-запрос всех точек за один вызов
            dists, idxs = tree.query(coords, k=self.n_neighbors)  # (N, k), (N, k)

            # 2. Получаем координаты и поля соседей для всех точек
            neighbour_coords = coarse_coords[idxs]  # (N, k, 3)
            neighbour_fields = coarse_fields[idxs]  # (N, k, 8)

            # 3. Масштаб: среднее расстояние для каждой точки
            scale = np.mean(dists, axis=1, keepdims=True) + 1e-8  # (N, 1)

            # 4. Относительные координаты
            rel_coords = (neighbour_coords - coords[:, np.newaxis, :]) / scale[..., np.newaxis]  # (N, k, 3)

            # 5. Формируем сырой патч (поля + относительные координаты)
            fields_flat = neighbour_fields.reshape(coords.shape[0], -1)  # (N, k*8)
            rel_flat = rel_coords.reshape(coords.shape[0], -1)  # (N, k*3)
            patch = np.concatenate([fields_flat, rel_flat], axis=1)  # (N, k*(8+3))

            # 6. Нормализация полей (если включена)
            if self.normalize:
                n_fields_flat = self.n_neighbors * 8
                # Разделяем поля и относительные координаты для нормализации
                patch_fields = patch[:, :n_fields_flat].reshape(coords.shape[0], self.n_neighbors, 8)
                patch_rel = patch[:, n_fields_flat:].reshape(coords.shape[0], self.n_neighbors, 3)
                # Нормализуем поля, используя self.fields_mean_np и self.fields_std_np
                patch_fields_norm = (patch_fields - self.fields_mean_np) / (self.fields_std_np + 1e-20)
                # Склеиваем обратно
                patch = np.concatenate([
                    patch_fields_norm.reshape(coords.shape[0], -1),
                    patch_rel.reshape(coords.shape[0], -1)
                ], axis=1)

            # Сохраняем как двумерный массив (N, patch_dim) — __getitem__ будет работать без изменений
            self.precomputed_patches.append(patch)

    def get_id_slice(self, id_: int) -> slice:
        idx = self.id_to_index.get(id_)
        if idx is None:
            return slice(0, 0)
        return slice(self.cumulative_sizes[idx], self.cumulative_sizes[idx + 1])

    def __len__(self):
        return self.total_points

    def __getitem__(self, idx):
        if self.total_points == 0:
            return {}
        id_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        local_idx = idx - self.cumulative_sizes[id_idx]
        id_ = int(self.df.iloc[id_idx]['id'])
        coords = self.coords_list[id_idx][local_idx]
        fields = self.fields_list[id_idx][local_idx]
        shape = np.array(self.shape_params[id_idx])

        if self.precomputed_patches is not None:
            patch = self.precomputed_patches[id_idx][local_idx]
        else:
            patch = np.zeros(self.n_neighbors * (8 + 3))

        if self.normalize:
            coords = (coords - self.coords_mean) / self.coords_std
            shape = (shape - self.shape_mean) / self.shape_std

        voltage_true = self.df[self.df['id'] == id_].iloc[0]['voltage_complex']

        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'shape_params': torch.tensor(shape, dtype=torch.float32),
            'coarse_patch': torch.tensor(patch, dtype=torch.float32),
            'target': torch.tensor(fields, dtype=torch.float32),
            'id': id_,
            'fields_mean': self.fields_mean_tensor,   # готовый тензор
            'fields_std': self.fields_std_tensor,     # готовый тензор
            'is_bottom': torch.tensor(self.bottom_mask_list[id_idx][local_idx].item(), dtype=torch.bool),
            'is_top': torch.tensor(self.top_mask_list[id_idx][local_idx].item(), dtype=torch.bool),
            'voltage_true': torch.tensor(abs(voltage_true), dtype=torch.float32)
        }


# ---------------------- Датасет для коллокационных точек ----------------------
class CollocationDataset(Dataset):
    def __init__(self,
                 ids: List[int],
                 shape_params: Dict[int, Tuple[float, float]],
                 coarse_data: Dict[int, Tuple[cKDTree, np.ndarray, np.ndarray]],
                 n_points_per_id: int = 200,
                 n_neighbors: int = 8,
                 normalize: bool = True,
                 coords_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 shape_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 fields_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.ids = ids
        self.n_points_per_id = n_points_per_id
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.total_points = len(ids) * n_points_per_id

        # Статистики
        self.coords_mean, self.coords_std = coords_stats if coords_stats else (np.zeros(3), np.ones(3))
        self.shape_mean, self.shape_std = shape_stats if shape_stats else (np.zeros(2), np.ones(2))
        self.fields_mean, self.fields_std = fields_stats if fields_stats else (np.zeros(8), np.ones(8))

        # Предвычисление констант для каждого ID
        self.id_info = {}
        for id_ in ids:
            r_um, h_um = shape_params[id_]
            R = r_um * 1e-6
            H = h_um * 1e-6
            z0 = -H / 2
            tree, coarse_coords, coarse_fields = coarse_data[id_]
            self.id_info[id_] = {
                'r_um': r_um,
                'h_um': h_um,
                'R': R,
                'H': H,
                'z0': z0,
                'tree': tree,
                'coarse_coords': coarse_coords,
                'coarse_fields': coarse_fields
            }

    def __len__(self):
        return self.total_points

    def __getitems__(self, indices):
        """
        Эффективно генерирует батч коллокационных точек.
        Группирует запросы по ID, использует batch-запросы к KDTree.
        """
        if not indices:
            return []

        # Группируем индексы по ID
        from collections import defaultdict
        id_to_positions = defaultdict(list)   # id -> [(position_in_batch, local_idx), ...]
        for pos, idx in enumerate(indices):
            id_idx = idx // self.n_points_per_id
            local_idx = idx % self.n_points_per_id
            id_ = self.ids[id_idx]
            id_to_positions[id_].append((pos, local_idx))

        results = [None] * len(indices)

        for id_, pos_list in id_to_positions.items():
            info = self.id_info[id_]
            r_um = info['r_um']
            h_um = info['h_um']
            R = info['R']
            H = info['H']
            z0 = info['z0']
            tree = info['tree']
            coarse_coords = info['coarse_coords']
            coarse_fields = info['coarse_fields']

            # Уникальные локальные индексы, для которых нужно сгенерировать точки
            unique_locals = sorted(set(local_idx for _, local_idx in pos_list))
            n_points = len(unique_locals)

            # 1. Векторизованная генерация точек в цилиндре
            u = np.random.random(n_points)
            v = np.random.random(n_points)
            w = np.random.random(n_points)
            r = R * np.sqrt(u)
            theta = 2 * np.pi * v
            z = z0 + H * w
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.stack([x, y, z], axis=1)            # (n_points, 3)

            # 2. Batch-запрос к KDTree
            dists, idxs = tree.query(points, k=self.n_neighbors)   # (n_points, k), (n_points, k)

            # 3. Извлечение данных соседей
            neighbor_coords = coarse_coords[idxs]           # (n_points, k, 3)
            neighbor_fields = coarse_fields[idxs]           # (n_points, k, 8)

            # 4. Масштаб и относительные координаты
            scale = np.mean(dists, axis=1, keepdims=True) + 1e-8   # (n_points, 1)
            rel_coords = (neighbor_coords - points[:, np.newaxis, :]) / scale[..., np.newaxis]  # (n_points, k, 3)

            # 5. Формирование патча
            fields_flat = neighbor_fields.reshape(n_points, -1)     # (n_points, k*8)
            rel_flat = rel_coords.reshape(n_points, -1)             # (n_points, k*3)
            patches = np.concatenate([fields_flat, rel_flat], axis=1)  # (n_points, k*(8+3))

            # 6. Нормализация (если нужно)
            points_norm = points
            if self.normalize:
                points_norm = (points - self.coords_mean) / self.coords_std
                # Нормализация полей внутри патча
                n_fields_flat = self.n_neighbors * 8
                patch_fields = patches[:, :n_fields_flat].reshape(n_points, self.n_neighbors, 8)
                patch_rel = patches[:, n_fields_flat:].reshape(n_points, self.n_neighbors, 3)
                patch_fields_norm = (patch_fields - self.fields_mean) / (self.fields_std + 1e-20)
                patches = np.concatenate([
                    patch_fields_norm.reshape(n_points, -1),
                    patch_rel.reshape(n_points, -1)
                ], axis=1)

            # 7. Параметры формы (одинаковы для всех точек данного ID)
            shape = np.array([r_um, h_um])
            if self.normalize:
                shape = (shape - self.shape_mean) / self.shape_std

            # 8. Сопоставление сгенерированных точек с исходными позициями
            # Создаём словарь local_idx -> (point, patch)
            local_map = {local_idx: (points_norm[i], patches[i]) for i, local_idx in enumerate(unique_locals)}
            for pos, local_idx in pos_list:
                point, patch = local_map[local_idx]
                results[pos] = {
                    'coords': torch.tensor(point, dtype=torch.float32),
                    'shape_params': torch.tensor(shape, dtype=torch.float32),
                    'coarse_patch': torch.tensor(patch, dtype=torch.float32),
                    'id': id_
                }

        return results

    def __getitem__(self, idx):
        """Обратная совместимость для DataLoader без поддержки __getitems__."""
        return self.__getitems__([idx])[0]


# ---------------------- Модель SR-PINN ----------------------
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
    def __init__(self,
                 n_spatial: int = 3,
                 n_shape_params: int = 2,
                 n_coarse_nodes: int = 8,
                 n_field_vars: int = 8,
                 hidden_dim: int = 256,
                 n_blocks: int = 6,
                 fourier_mapping_size: int = 128,
                 fourier_scale: float = 5.0,
                 output_scale_init=100.0):
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
        self.output_scale = nn.Parameter(torch.tensor(output_scale_init, dtype=torch.float32))
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
        out = out * self.output_scale
        return out


class StressPINNLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0, lambda_voltage: float = 10.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_voltage = lambda_voltage
        self.mse = nn.MSELoss()

    def forward(self, model, batch, batch_pde=None):
        device = next(model.parameters()).device

        # Один forward для всего батча
        coords = batch['coords']
        shape = batch['shape_params']
        patch = batch['coarse_patch']
        pred_norm = model(coords, shape, patch)  # (N, 8)

        loss_data = torch.tensor(0.0, device=device)
        if 'target' in batch:
            loss_data = self.mse(pred_norm, batch['target'])

        loss_voltage = torch.tensor(0.0, device=device)
        if 'is_top' in batch and 'is_bottom' in batch and 'id' in batch:
            fields_mean = batch['fields_mean']
            fields_std = batch['fields_std']
            # Денормализуем phi (индексы 6,7)
            phi_real = pred_norm[:, 6] * fields_std[..., 6] + fields_mean[..., 6]
            phi_imag = pred_norm[:, 7] * fields_std[..., 7] + fields_mean[..., 7]
            phi = torch.complex(phi_real, phi_imag)

            ids = batch['id']
            is_top = batch['is_top']
            is_bottom = batch['is_bottom']
            unique_ids = torch.unique(ids)
            for uid in unique_ids:
                mask = (ids == uid)
                top_mask = mask & is_top
                bottom_mask = mask & is_bottom
                if top_mask.any() and bottom_mask.any():
                    phi_top = phi[top_mask].mean()
                    phi_bottom = phi[bottom_mask].mean()
                    V_pred = phi_top - phi_bottom
                    V_true = batch['voltage_true'][mask].mean()  # или .mean(), так как одинаково
                    loss_voltage += self.mse(V_pred.abs(), V_true)

        total_loss = self.lambda_data * loss_data + self.lambda_voltage * loss_voltage
        return total_loss, {
            'loss_data': loss_data.item(),
            'loss_voltage': loss_voltage.item(),
            'total_loss': total_loss.item()
        }


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

    fine_df = load_all_csv_cached(data_dir, 'results_fine', 'fine_df')
    coarse_df = load_all_csv_cached(data_dir, 'results_coarse', 'coarse_df')

    if fine_df.empty:
        raise ValueError("Не найдены results_fine*.csv")

    # === 1. Статистика полей — ТОЛЬКО ПО FINE-ДАННЫМ (целевая правда) ===
    fields_mean, fields_std = compute_stats_incremental(generate_fine_fields(train_ids, data_dir))

    print(f"Fields stats FROM FINE data: mean[6]={fields_mean[6]:.3e}, "
          f"std[6]={fields_std[6]:.3e} | mean[7]={fields_mean[7]:.3e}, std[7]={fields_std[7]:.3e}")

    external_stats = (fields_mean, fields_std)

    # === 2. Создаём train и val датасеты ===
    train_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=train_ids,
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True,
        external_stats=external_stats,
        subsample_ratio=1.0  # ускорение обучения
    )

    val_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=val_ids,
        mesh_type='fine',
        n_neighbors=n_neighbors,
        normalize=True,
        external_stats=external_stats,
        subsample_ratio=1.0  # полная валидация
    )

    # === 3. Загружаем coarse-данные только для train + val ===
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

    # Привязываем coarse к датасетам
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

    # === 4. Collocation dataset (только для train) ===

    # Строим словари для быстрого доступа
    coarse_shape_dict = {
        row['id']: (row['r_um'], row['h_um'])
        for _, row in coarse_df.iterrows()
    }
    coarse_id_to_index = {id_: idx for idx, id_ in enumerate(coarse_id_list)}

    # Кэш деревьев (строим один раз)
    coarse_trees_cache = {}
    for idx, id_ in enumerate(coarse_id_list):
        # build_kdtree – ваша функция, обёртка над cKDTree
        coarse_trees_cache[id_] = build_kdtree(coarse_coords_list[idx])

    # Создаём colloc_dataset без повторного построения
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

    # === 5. Статистики для сохранения ===
    stats = {
        'coords_mean': train_dataset.coords_mean,
        'coords_std': train_dataset.coords_std,
        'shape_mean': train_dataset.shape_mean,
        'shape_std': train_dataset.shape_std,
        'fields_mean': torch.tensor(fields_mean, dtype=torch.float32),
        'fields_std': torch.tensor(fields_std, dtype=torch.float32)
    }

    return train_dataset, val_dataset, colloc_dataset, stats


# ---------------------- Вычисление ошибки напряжения ----------------------
def compute_voltage_error(model, val_dataset, device, verbose=False, return_list=False, eps=1e-20):
    model.eval()
    ids_val = val_dataset.df['id'].values if hasattr(val_dataset, 'df') else []
    errors = []
    with torch.no_grad():
        for id_ in ids_val:
            slc = val_dataset.get_id_slice(id_)
            if slc.stop - slc.start == 0:
                continue
            indices = list(range(slc.start, slc.stop))
            batch_coords = []
            batch_shape = []
            batch_patch = []
            for idx in indices:
                item = val_dataset[idx]
                batch_coords.append(item['coords'].unsqueeze(0))
                batch_shape.append(item['shape_params'].unsqueeze(0))
                batch_patch.append(item['coarse_patch'].unsqueeze(0))
            if not batch_coords:
                continue
            coords = torch.cat(batch_coords, dim=0).to(device)
            shape = torch.cat(batch_shape, dim=0).to(device)
            patch = torch.cat(batch_patch, dim=0).to(device)
            pred_fields = model(coords, shape, patch).cpu().numpy()
            fields_mean = val_dataset.fields_mean_np
            fields_std = val_dataset.fields_std_np
            pred_fields_denorm = pred_fields * fields_std + fields_mean
            phi_pred = pred_fields_denorm[:, 6] + 1j * pred_fields_denorm[:, 7]
            id_idx = val_dataset.id_to_index[id_]
            bottom_mask = val_dataset.bottom_mask_list[id_idx]
            top_mask = val_dataset.top_mask_list[id_idx]
            if not np.any(bottom_mask) or not np.any(top_mask):
                continue
            phi_bottom = phi_pred[bottom_mask].mean()
            phi_top = phi_pred[top_mask].mean()
            V_pred = phi_top - phi_bottom
            row = val_dataset.df[val_dataset.df['id'] == id_].iloc[0]
            V_true = row['voltage_complex']
            # Логарифмическая ошибка для модуля напряжения
            abs_pred = abs(V_pred)
            abs_true = abs(V_true)
            # Добавляем eps, чтобы избежать log(0)
            #log_pred = np.log10(abs_pred + eps)
            #log_true = np.log10(abs_true + eps)
            #error = abs(log_pred - log_true)   # ошибка в декадах
            error = abs(V_pred - V_true) / abs_true
            errors.append(error)
            if verbose:
                print(f"ID {id_}: V_pred = {V_pred:.3e}, V_true = {V_true:.3e}")
                #print(f"  phi_top = {p:.3f}, phi_bottom = {abs_true:.3f}")
                print(f"  relative error = {error:.4f}")
    if return_list:
        return np.array(errors)
    return np.mean(errors) if errors else float('inf')

# ---------------------- Обучение ----------------------
def train_srpinn(model, train_loader, val_dataset, colloc_loader, n_epochs, device, lr=5e-4, pde_every=5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = StressPINNLoss(lambda_data=1.0, lambda_voltage=10.0)
    best_voltage_error = float('inf')
    colloc_iter = iter(colloc_loader)
    step_counter = 0

    # Создаём val_loader один раз
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Ключи, которые нужно переносить на device для train и val
    train_keys = {'coords', 'shape_params', 'coarse_patch', 'target',
                  'voltage_true', 'is_bottom', 'is_top', 'fields_mean', 'fields_std', 'id'}
    colloc_keys = {'coords', 'shape_params', 'coarse_patch'}

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            compute_pde = (step_counter % pde_every == 0)
            step_counter += 1

            if compute_pde:
                try:
                    batch_pde = next(colloc_iter)
                except StopIteration:
                    colloc_iter = iter(colloc_loader)
                    batch_pde = next(colloc_iter)
                batch_pde = to_device(batch_pde, device)
            else:
                batch_pde = None

            batch = to_device(batch, device)

            optimizer.zero_grad()
            loss, loss_dict = criterion(model, batch, batch_pde)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss_dict)

        scheduler.step()

        # Валидация (не каждую эпоху)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = to_device(vbatch, device)
                    _, loss_dict = criterion(model, vbatch, None)
                    val_losses.append(loss_dict)
            avg_val_loss = np.mean([d['total_loss'] for d in val_losses])

            voltage_error = compute_voltage_error(model, val_dataset, device, verbose=True)
            avg_train_loss = np.mean([d['total_loss'] for d in train_losses])

            print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, "
                  f"val_loss={avg_val_loss:.6f}, voltage_rel_error={voltage_error:.4f}")

            if voltage_error < best_voltage_error:
                best_voltage_error = voltage_error
                torch.save(model.state_dict(), 'best_srpinn_model_voltage.pth')
                print(f"  >>> Saved best model (voltage error {best_voltage_error:.4f}) <<<")

def parse_idx_train(data_dir: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Parse directory contents to extract numeric IDs from files matching the pattern
    '*id_<number>_fine*.mat'. The IDs are sorted and split into train (60%),
    validation (20%) and test (20%) sets.

    Args:
        data_dir: Path to the directory containing the files.

    Returns:
        A tuple of three lists: (train_ids, val_ids, test_ids).
    """

    directory_contents = os.listdir(data_dir)
    pattern = re.compile(r'id_(\d+)_fine.*\.mat$')

    ids = []
    for file in directory_contents:
        match = pattern.search(file)
        if match:
            ids.append(int(match.group(1)))

    ids_sorted = sorted(set(ids))

    n = len(ids_sorted)

    test_split = n // 5
    val_split = 2 * n // 5

    test_ids = ids_sorted[:test_split]
    val_ids = ids_sorted[test_split:val_split]
    train_ids = ids_sorted[val_split:]

    return train_ids, val_ids, test_ids


# ---------------------- Запуск ----------------------
if __name__ == "__main__":
    data_dir = path_to_files
    coarse_ids = list(range(1, 101))

    train_ids, val_ids, test_ids = parse_idx_train(data_dir)

    print(f"Train IDs: {train_ids}")
    print(f"Val   IDs: {val_ids}")
    print(f"Test  IDs: {test_ids}")

    # Подготовка (теперь с отдельными train/val)
    train_dataset, val_dataset, colloc_dataset, stats = prepare_datasets(
        data_dir, coarse_ids, train_ids, val_ids, n_neighbors=8
    )

    # Test dataset (полный)
    fine_df = load_all_csv(data_dir, 'results_fine')
    fine_df['voltage_complex'] = fine_df['voltage'].apply(parse_complex)
    test_dataset = CylinderStressDataset(
        data_dir=data_dir,
        csv_df=fine_df,
        ids=test_ids,
        mesh_type='fine',
        n_neighbors=8,
        normalize=True,
        external_stats=(stats['fields_mean'], stats['fields_std']),
        subsample_ratio=1.0
    )

    # Coarse для test
    coarse_coords_dict = {}
    coarse_fields_dict = {}
    for id_ in test_ids:
        try:
            coords, fields = load_mat_with_cache(id_, "coarse", data_dir)
            coarse_coords_dict[id_] = coords
            coarse_fields_dict[id_] = fields
        except FileNotFoundError:
            print(f"Warning: coarse file for ID {id_} not found, skipping")
            continue

    filtered = [(id_, coarse_coords_dict[id_], coarse_fields_dict[id_])
                for id_ in test_ids if id_ in coarse_coords_dict]
    if filtered:
        test_coarse_ids, test_coarse_coords, test_coarse_fields = zip(*filtered)
        test_dataset.set_coarse_data(list(test_coarse_coords), list(test_coarse_fields), list(test_coarse_ids))

    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    colloc_loader = DataLoader(colloc_dataset, batch_size=64, shuffle=True)

    # Модель
    model = SRPINN()
    model = torch.compile(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Обучение
    train_srpinn(model, train_loader, val_dataset, colloc_loader,
                 n_epochs=300, device=device, lr=5e-4, pde_every=5)

    # Тест
    print("\n=== ОЦЕНКА НА ТЕСТОВЫХ ID ===")
    model.load_state_dict(torch.load('best_srpinn_model_voltage.pth', map_location=device))
    model.eval()
    test_errors = compute_voltage_error(model, test_dataset, device, verbose=True, return_list=True)
    print(f"Test mean log error (decades): {np.mean(test_errors):.4f}")
    print(f"Test median log error: {np.median(test_errors):.4f}")

    # Сохранение с статистиками
    torch.save({
        'model_state_dict': model.state_dict(),
        'coords_mean': stats['coords_mean'],
        'coords_std': stats['coords_std'],
        'shape_mean': stats['shape_mean'],
        'shape_std': stats['shape_std'],
        'fields_mean': stats['fields_mean'],
        'fields_std': stats['fields_std']
    }, 'best_srpinn_model_with_stats.pth')

    print("Модель сохранена: best_srpinn_model_with_stats.pth")
