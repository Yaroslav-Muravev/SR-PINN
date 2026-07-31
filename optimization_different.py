import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import glob
import re

# ---------------------- Загрузка coarse-данных ----------------------
def parse_complex(s):
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

# ---------------------- Подготовка интерполятора ----------------------
data_dir = "./files/"
df = load_coarse_voltage_data(data_dir)
R = df['r_um'].values
H = df['h_um'].values
V = df['V_abs'].values

points = np.column_stack([R, H])
values = V
rbf = RBFInterpolator(points, values, kernel='multiquadric', epsilon=1.0)

# Целевая функция (для минимизации отрицательного значения)
def objective(x):
    r, h = x[0], x[1]
    # ограничим область допустимых значений (не выходить за пределы данных)
    if r < R.min() or r > R.max() or h < H.min() or h > H.max():
        return 1e10
    val = rbf([[r, h]])[0]
    return -val

# Начальная точка (например, середина области)
x0 = np.array([(R.min() + R.max())/2, (H.min() + H.max())/2])
print(f"Начальная точка: R={x0[0]:.2f}, H={x0[1]:.2f}")

# История для траектории
history = [x0.copy()]

# Функция callback для записи шагов
def callback(xk):
    history.append(xk.copy())

# Запуск оптимизации методом Nelder-Mead (не требует градиента)
result = minimize(objective, x0, method='Nelder-Mead', bounds=[(R.min(), R.max()), (H.min(), H.max())],
                  options={'maxiter': 30, 'disp': True}, callback=callback)

opt_R, opt_H = result.x
opt_V = -result.fun
print(f"Оптимум: R={opt_R:.2f} мкм, H={opt_H:.2f} мкм, |V|={opt_V:.3e} В")
print(f"Количество итераций: {len(history)-1}")

# Если история пуста, добавим начальную точку
if len(history) == 0:
    history.append(x0)
history = np.array(history)

# ---------------------- Построение ландшафта и траектории ----------------------
R_grid = np.linspace(R.min(), R.max(), 50)
H_grid = np.linspace(H.min(), H.max(), 50)
R_mesh, H_mesh = np.meshgrid(R_grid, H_grid)
V_mesh = rbf(np.column_stack([R_mesh.ravel(), H_mesh.ravel()])).reshape(R_mesh.shape)

# 3D график
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(R_mesh, H_mesh, V_mesh, cmap='plasma', alpha=0.7, edgecolor='none')
ax.scatter(R, H, V, c='black', s=20, alpha=0.5, label='Coarse данные')

if len(history) > 0:
    # Значения V в точках траектории
    V_hist = [rbf([[pt[0], pt[1]]])[0] for pt in history]
    ax.plot(history[:,0], history[:,1], V_hist, 'o-', color='lime', linewidth=2, markersize=6, label='Траектория')
    ax.scatter(history[0,0], history[0,1], V_hist[0], color='blue', s=100, marker='s', label='Старт')
    ax.scatter(opt_R, opt_H, opt_V, color='red', s=150, marker='*', label='Оптимум')

ax.set_xlabel('R (мкм)')
ax.set_ylabel('H (мкм)')
ax.set_zlabel('|V| (В)')
ax.set_title('Оптимизация формы цилиндра (максимизация |V|)')
ax.legend()
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='|V| (В)')
plt.savefig('optimization_trajectory_3d.png', dpi=150)
plt.show()

# 2D контур
plt.figure(figsize=(10, 8))
contour = plt.contourf(R_mesh, H_mesh, V_mesh, levels=50, cmap='plasma')
plt.colorbar(contour, label='|V| (В)')
plt.scatter(R, H, c='black', s=20, alpha=0.5, label='Coarse точки')
if len(history) > 0:
    plt.plot(history[:,0], history[:,1], 'o-', color='lime', linewidth=2, markersize=6, label='Траектория')
    plt.scatter(history[0,0], history[0,1], color='blue', s=100, marker='s', label='Старт')
    plt.scatter(opt_R, opt_H, color='red', s=150, marker='*', label='Оптимум')
plt.xlabel('R (мкм)')
plt.ylabel('H (мкм)')
plt.title('Контурная карта |V| с траекторией оптимизации')
plt.legend()
plt.savefig('optimization_trajectory_2d.png', dpi=150)
plt.show()