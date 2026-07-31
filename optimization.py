import numpy as np
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, griddata
from mpl_toolkits.mplot3d import Axes3D

# ---------------------- Вспомогательные функции ----------------------
def parse_complex(s):
    """Парсит строку напряжения в комплексное число."""
    if isinstance(s, (int, float)):
        return complex(s, 0)
    s = str(s).strip().replace(' ', '')
    pattern = r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?([-+]\d*\.?\d+(?:[eE][-+]?\d+)?)i$'
    m = re.match(pattern, s)
    if m:
        re_part = float(m.group(1)) if m.group(1) else 0.0
        im_part = float(m.group(2))
        return complex(re_part, im_part)
    try:
        return complex(s)
    except:
        return complex(np.nan, np.nan)

def load_coarse_voltage_data(data_dir="./files/"):
    """Загружает все results_coarse*.csv, возвращает DataFrame с r_um, h_um, |V|."""
    csv_files = glob.glob(data_dir + "results_coarse*.csv")
    if not csv_files:
        raise FileNotFoundError("Не найдены results_coarse*.csv в папке ./files/")
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Парсим voltage
        df['voltage_complex'] = df['voltage'].apply(parse_complex)
        df['V_abs'] = df['voltage_complex'].apply(abs)
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True)
    # Убираем строки с NaN
    df_all = df_all.dropna(subset=['r_um', 'h_um', 'V_abs'])
    return df_all[['id', 'r_um', 'h_um', 'V_abs']]

# ---------------------- Загрузка данных ----------------------
data_dir = "./files/"
df = load_coarse_voltage_data(data_dir)
print(f"Загружено {len(df)} точек с coarse-расчётов.")
print(df.head())

# Извлекаем координаты и целевую величину
R = df['r_um'].values
H = df['h_um'].values
V = df['V_abs'].values

# ---------------------- Интерполяция поверхности ----------------------
# Создаём регулярную сетку для отображения
R_grid = np.linspace(R.min(), R.max(), 50)
H_grid = np.linspace(H.min(), H.max(), 50)
R_mesh, H_mesh = np.meshgrid(R_grid, H_grid)

# Используем RBF интерполяцию (гладкая)
# Объединяем входные точки в 2D
points = np.column_stack([R, H])
values = V

# RBF с многоквадрическим ядром
rbf = RBFInterpolator(points, values, kernel='multiquadric', epsilon=1.0)
V_interp = rbf(np.column_stack([R_mesh.ravel(), H_mesh.ravel()])).reshape(R_mesh.shape)

# Находим максимум на интерполированной поверхности
max_idx = np.argmax(V_interp)
max_R = R_mesh.ravel()[max_idx]
max_H = H_mesh.ravel()[max_idx]
max_V = V_interp.ravel()[max_idx]

print(f"Максимум |V| на интерполированной поверхности: {max_V:.3e} при R={max_R:.2f} мкм, H={max_H:.2f} мкм")

# Находим максимум среди исходных точек
idx_max_raw = np.argmax(V)
print(f"Максимум |V| среди имеющихся coarse-точек: {V[idx_max_raw]:.3e} при R={R[idx_max_raw]:.2f} мкм, H={H[idx_max_raw]:.2f} мкм (ID={df.iloc[idx_max_raw]['id']})")

# ---------------------- 3D визуализация ----------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Исходные точки (scatter)
sc = ax.scatter(R, H, V, c=V, cmap='viridis', s=50, alpha=0.8, label='Coarse данные')

# Интерполированная поверхность
surf = ax.plot_surface(R_mesh, H_mesh, V_interp, cmap='plasma', alpha=0.6, edgecolor='none')

# Отметим максимум на поверхности
ax.scatter([max_R], [max_H], [max_V], color='red', s=100, marker='*', label=f'Максимум (интерп.)\nR={max_R:.1f}, H={max_H:.1f}, |V|={max_V:.2e}')

# Отметим максимум среди исходных точек
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