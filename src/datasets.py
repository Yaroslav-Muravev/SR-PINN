import torch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
import os
import re
from typing import List, Optional, Tuple, Dict

from data_utils import (
    load_mat_with_cache,
    compute_stats_incremental,
    parse_complex,
    build_kdtree,
)


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
            eps_z = 1e-4 * (z_max - z_min)
            bottom_mask = np.abs(z_vals - z_min) < eps_z
            top_mask = np.abs(z_vals - z_max) < eps_z

            if subsample_ratio < 1.0:
                idx_bottom = np.where(bottom_mask)[0]
                idx_top = np.where(top_mask)[0]
                idx_boundary = np.concatenate([idx_bottom, idx_top])
                idx_inner = np.where(~(bottom_mask | top_mask))[0]
                keep = list(idx_boundary)
                n_inner = len(idx_inner)
                n_inner_keep = int(n_inner * subsample_ratio)
                if n_inner_keep > 0:
                    idx_inner_keep = np.random.choice(idx_inner, n_inner_keep, replace=False)
                    keep.extend(idx_inner_keep)
                keep = np.sort(keep)
                coords = coords[keep]
                fields = fields[keep]
                bottom_mask = bottom_mask[keep]
                top_mask = top_mask[keep]

            self.coords_list.append(coords)
            self.fields_list.append(fields)
            self.shape_params.append([r_um, h_um])
            self.id_to_index[id_] = len(self.coords_list) - 1
            self.bottom_mask_list.append(bottom_mask)
            self.top_mask_list.append(top_mask)
            self.total_points += coords.shape[0]
            self.cumulative_sizes.append(self.total_points)

        self.n_field_vars = fields.shape[1]

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

            self.fields_mean_tensor = torch.from_numpy(self.fields_mean_np).float()
            self.fields_std_tensor = torch.from_numpy(self.fields_std_np).float()
        else:
            self.coords_mean = np.zeros(3)
            self.coords_std = np.ones(3)
            self.shape_mean = np.zeros(2)
            self.shape_std = np.ones(2)
            if self.total_points > 0:
                dim = self.fields_list[0].shape[1]
            else:
                dim = 7
            self.fields_mean_np = np.zeros(dim)
            self.fields_std_np = np.ones(dim)
            self.fields_mean_tensor = torch.zeros(dim, dtype=torch.float32)
            self.fields_std_tensor = torch.ones(dim, dtype=torch.float32)

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
        if self.coarse_trees is None:
            self.precomputed_patches = None
            return

        self.precomputed_patches = []
        for id_idx, coords in enumerate(self.coords_list):
            id_ = int(self.df.iloc[id_idx]['id'])
            if id_ not in self.coarse_trees:
                patches = np.zeros((coords.shape[0], self.n_neighbors * (self.n_field_vars + 3)), dtype=np.float32)
                self.precomputed_patches.append(patches)
                continue

            tree = self.coarse_trees[id_]
            coarse_coords = self.coarse_coords[id_]
            coarse_fields = self.coarse_fields[id_]

            dists, idxs = tree.query(coords, k=self.n_neighbors)
            neighbour_coords = coarse_coords[idxs]
            neighbour_fields = coarse_fields[idxs]
            scale = np.mean(dists, axis=1, keepdims=True) + 1e-8
            rel_coords = (neighbour_coords - coords[:, np.newaxis, :]) / scale[..., np.newaxis]
            fields_flat = neighbour_fields.reshape(coords.shape[0], -1)
            rel_flat = rel_coords.reshape(coords.shape[0], -1)
            patch = np.concatenate([fields_flat, rel_flat], axis=1)

            if self.normalize:
                n_fields_flat = self.n_neighbors * self.n_field_vars
                patch_fields = patch[:, :n_fields_flat].reshape(coords.shape[0], self.n_neighbors, self.n_field_vars)
                patch_rel = patch[:, n_fields_flat:].reshape(coords.shape[0], self.n_neighbors, 3)
                patch_fields_norm = (patch_fields - self.fields_mean_np) / (self.fields_std_np + 1e-20)
                patch = np.concatenate([
                    patch_fields_norm.reshape(coords.shape[0], -1),
                    patch_rel.reshape(coords.shape[0], -1)
                ], axis=1)

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
            patch = np.zeros(self.n_neighbors * (self.n_field_vars + 3))

        if self.normalize:
            coords = (coords - self.coords_mean) / self.coords_std
            shape = (shape - self.shape_mean) / self.shape_std
            fields = (fields - self.fields_mean_np) / self.fields_std_np

        voltage_true = self.df[self.df['id'] == id_].iloc[0]['voltage_complex']

        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'shape_params': torch.tensor(shape, dtype=torch.float32),
            'coarse_patch': torch.tensor(patch, dtype=torch.float32),
            'target': torch.tensor(fields, dtype=torch.float32),
            'id': id_,
            'fields_mean': self.fields_mean_tensor,
            'fields_std': self.fields_std_tensor,
            'is_bottom': torch.tensor(self.bottom_mask_list[id_idx][local_idx].item(), dtype=torch.bool),
            'is_top': torch.tensor(self.top_mask_list[id_idx][local_idx].item(), dtype=torch.bool),
            'voltage_true': torch.tensor(abs(voltage_true), dtype=torch.float32)
        }

    def get_boundary_batch(self, id_: int):
        idx = self.id_to_index[id_]
        bottom_mask = self.bottom_mask_list[idx]
        top_mask = self.top_mask_list[idx]
        boundary_mask = bottom_mask | top_mask
        boundary_indices = np.where(boundary_mask)[0]

        if len(boundary_indices) == 0:
            raise RuntimeError(f"No boundary points found for ID {id_}. Check eps_z or mesh.")

        coords = self.coords_list[idx][boundary_indices]
        fields = self.fields_list[idx][boundary_indices]
        shape = np.array(self.shape_params[idx])
        if self.precomputed_patches is not None:
            patches = self.precomputed_patches[idx][boundary_indices]
        else:
            patches = np.zeros((len(boundary_indices), self.n_neighbors * (self.n_field_vars + 3)))

        if self.normalize:
            coords = (coords - self.coords_mean) / self.coords_std
            shape = (shape - self.shape_mean) / self.shape_std
            fields = (fields - self.fields_mean_np) / self.fields_std_np

        bottom_mask_sub = bottom_mask[boundary_indices]
        top_mask_sub = top_mask[boundary_indices]
        voltage_true = self.df[self.df['id'] == id_].iloc[0]['voltage_complex']

        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'shape_params': torch.tensor(shape, dtype=torch.float32).unsqueeze(0).repeat(len(boundary_indices), 1),
            'coarse_patch': torch.tensor(patches, dtype=torch.float32),
            'target': torch.tensor(fields, dtype=torch.float32),
            'id': torch.tensor([id_] * len(boundary_indices)),
            'fields_mean': self.fields_mean_tensor,
            'fields_std': self.fields_std_tensor,
            'is_bottom': torch.tensor(bottom_mask_sub, dtype=torch.bool),
            'is_top': torch.tensor(top_mask_sub, dtype=torch.bool),
            'voltage_true': torch.tensor(abs(voltage_true), dtype=torch.float32).repeat(len(boundary_indices))
        }


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

        self.coords_mean, self.coords_std = coords_stats if coords_stats else (np.zeros(3), np.ones(3))
        self.shape_mean, self.shape_std = shape_stats if shape_stats else (np.zeros(2), np.ones(2))
        self.fields_mean, self.fields_std = fields_stats if fields_stats else (np.zeros(8), np.ones(8))

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

        self.n_field_vars = coarse_fields.shape[1]

    def __len__(self):
        return self.total_points

    def __getitems__(self, indices):
        if not indices:
            return []

        from collections import defaultdict
        id_to_positions = defaultdict(list)
        for pos, idx in enumerate(indices):
            id_idx = idx // self.n_points_per_id
            local_idx = idx % self.n_points_per_id
            id_ = self.ids[id_idx]
            id_to_positions[id_].append((pos, local_idx))

        results = [None] * len(indices)

        for id_, pos_list in id_to_positions.items():
            info = self.id_info[id_]
            R = info['R']
            H = info['H']
            z0 = info['z0']
            tree = info['tree']
            coarse_coords = info['coarse_coords']
            coarse_fields = info['coarse_fields']

            unique_locals = sorted(set(local_idx for _, local_idx in pos_list))
            n_points = len(unique_locals)

            u = np.random.random(n_points)
            v = np.random.random(n_points)
            w = np.random.random(n_points)
            r = R * np.sqrt(u)
            theta = 2 * np.pi * v
            z = z0 + H * w
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.stack([x, y, z], axis=1)

            dists, idxs = tree.query(points, k=self.n_neighbors)
            neighbor_coords = coarse_coords[idxs]
            neighbor_fields = coarse_fields[idxs]
            scale = np.mean(dists, axis=1, keepdims=True) + 1e-8
            rel_coords = (neighbor_coords - points[:, np.newaxis, :]) / scale[..., np.newaxis]

            fields_flat = neighbor_fields.reshape(n_points, -1)
            rel_flat = rel_coords.reshape(n_points, -1)
            patches = np.concatenate([fields_flat, rel_flat], axis=1)

            points_norm = points
            if self.normalize:
                points_norm = (points - self.coords_mean) / self.coords_std
                n_fields_flat = self.n_neighbors * self.n_field_vars
                patch_fields = patches[:, :n_fields_flat].reshape(n_points, self.n_neighbors, self.n_field_vars)
                patch_rel = patches[:, n_fields_flat:].reshape(n_points, self.n_neighbors, 3)
                patch_fields_norm = (patch_fields - self.fields_mean) / (self.fields_std + 1e-20)
                patches = np.concatenate([
                    patch_fields_norm.reshape(n_points, -1),
                    patch_rel.reshape(n_points, -1)
                ], axis=1)

            shape = np.array([info['r_um'], info['h_um']])
            if self.normalize:
                shape = (shape - self.shape_mean) / self.shape_std

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
        return self.__getitems__([idx])[0]


class VoltageDataset(Dataset):
    def __init__(self, dataset: CylinderStressDataset, ids: List[int]):
        self.dataset = dataset
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        return self.dataset.get_boundary_batch(id_)


def parse_idx_train(data_dir: str) -> Tuple[List[int], List[int], List[int]]:
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