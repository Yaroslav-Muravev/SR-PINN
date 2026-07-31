import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

from src.data_utils import load_coarse_voltage_data
from src.config import PATH_TO_FILES


df = load_coarse_voltage_data(PATH_TO_FILES)
print(f"Загружено {len(df)} точек с coarse-расчётов.")
print(df.head())

R = df['r_um'].values
H = df['h_um'].values
V = df['V_abs'].values

R_grid = np.linspace(R.min(), R.max(), 50)
H_grid = np.linspace(H.min(), H.max(), 50)
R_mesh, H_mesh = np.meshgrid(R_grid, H_grid)

points = np.column_stack([R, H])
values = V

rbf = RBFInterpolator(points, values, kernel='multiquadric', epsilon=1.0)
V_interp = rbf(np.column_stack([R_mesh.ravel(), H_mesh.ravel()])).reshape(R_mesh.shape)

max_idx = np.argmax(V_interp)
max_R = R_mesh.ravel()[max_idx]
max_H = H_mesh.ravel()[max_idx]
max_V = V_interp.ravel()[max_idx]

print(f"Максимум |V| на интерполированной поверхности: {max_V:.3e} при R={max_R:.2f} мкм, H={max_H:.2f} мкм")

idx_max_raw = np.argmax(V)
print(f"Максимум |V| среди имеющихся coarse-точек: {V[idx_max_raw]:.3e} при R={R[idx_max_raw]:.2f} мкм, H={H[idx_max_raw]:.2f} мкм (ID={df.iloc[idx_max_raw]['id']})")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(R, H, V, c=V, cmap='viridis', s=50, alpha=0.8, label='Coarse данные')

surf = ax.plot_surface(R_mesh, H_mesh, V_interp, cmap='plasma', alpha=0.6, edgecolor='none')

ax.scatter([max_R], [max_H], [max_V], color='red', s=100, marker='*', label=f'Максимум (интерп.)\nR={max_R:.1f}, H={max_H:.1f}, |V|={max_V:.2e}')

ax.scatter([R[idx_max_raw]], [H[idx_max_raw]], [V[idx_max_raw]], color='orange', s=100, marker='o', edgecolor='black', label=f'Макс. coarse\nID={df.iloc[idx_max_raw]["id"]}')

ax.set_xlabel('R (мкм)')
ax.set_ylabel('H (мкм)')
ax.set_zlabel('|V| (В)')
ax.set_title('Ландшафт среднего по частоте в рабочем диапазоне напряжения |V| в зависимости от формы цилиндра')
ax.legend()
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='|V| (В)')
plt.tight_layout()
plt.savefig('voltage_landscape_coarse.png', dpi=150)
plt.show()
