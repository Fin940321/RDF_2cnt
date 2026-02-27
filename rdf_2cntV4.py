from MDAnalysis import *
import MDAnalysis.analysis.distances as distances
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from numba import njit, prange
from tqdm import tqdm

# ==================== Numba 加速函數 ====================
@njit(parallel=True, cache=True)
def compute_environment_labels(contact_dist_0, contact_dist_1, contact_dist_2, 
                                r_contact_0, r_contact_1, r_contact_2):
    """
    使用 Numba 加速環境標籤計算
    環境分類：
    - env=2: CNT only
    - env=3: Graphene/CNT 交界處
    """
    n = len(contact_dist_0)
    environment = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        # Graphene contact (electrode 0)
        close_0 = 1.0 if contact_dist_0[i] < r_contact_0 else 0.0
        # CNT contacts (electrodes 1 & 2)
        close_1 = 1.0 if contact_dist_1[i] < r_contact_1 else 0.0
        close_2 = 1.0 if contact_dist_2[i] < r_contact_2 else 0.0
        close_cnt = 1.0 if (close_1 + close_2) > 0 else 0.0
        
        # environment = close_0 * 1 + close_cnt * 2
        environment[i] = close_0 * 1.0 + close_cnt * 2.0
    
    return environment


@njit(cache=True)
def classify_atoms_by_environment(environment, n_atoms):
    """
    將原子依據環境標籤分類，回傳各環境的原子索引
    """
    # 預先計算各環境的原子數量
    count_env0 = 0  # CNT (env=2)
    count_env1 = 0  # Junction (env=3)
    
    for i in range(n_atoms):
        if environment[i] == 2.0:
            count_env0 += 1
        elif environment[i] == 3.0:
            count_env1 += 1
    
    # 建立索引陣列
    indices_env0 = np.empty(count_env0, dtype=np.int64)
    indices_env1 = np.empty(count_env1, dtype=np.int64)
    
    idx0 = 0
    idx1 = 0
    for i in range(n_atoms):
        if environment[i] == 2.0:
            indices_env0[idx0] = i
            idx0 += 1
        elif environment[i] == 3.0:
            indices_env1[idx1] = i
            idx1 += 1
    
    return indices_env0, indices_env1


@njit(parallel=True, cache=True)
def accumulate_histogram(rdf_count, hist_count, n_bins):
    """
    加速 histogram 累加
    """
    for i in prange(n_bins):
        rdf_count[i] += hist_count[i]
    return rdf_count


topology = "start_drudes.pdb"
trajectory = "FV_NVT.dcd"
u = Universe(topology, trajectory)

# --- 參數設定區 ---
r_electrode_contact = np.array([6.0, 6.0, 6.0, 6.0]) # 記得依據您的一維密度圖波谷來修改！
z_max = 130 # 已根據您的系統修正
rdf_max = 15
n_bins = 75
dr = float(rdf_max) / float(n_bins)
framestart = 5000
frameend = len(u.trajectory) - 1

# --- 1. 使用字典定義所有的基團 (Functional Groups) ---
# 將多個原子名稱直接寫在同一個字串中，MDAnalysis 會自動視為聯集 (Union)
selections = {
    'Tf2_polar': "resname Tf2 and name Otf Otf1 Otf2 Otf3",
    'Tf2_nonpolar': "resname Tf2 and name Ctf Ctf1",
    'BMI_polar': "resname BMI and name C1 C2 C21",
    'BMI_nonpolar': "resname BMI and name C3 C4 C5 C51 C6"
}

# 電極環境定義 (Segids)
classifier = ["segid A", "segid B", "segid C", "segid D", "segid E"]
electrode_groups = [u.select_atoms(elec) for elec in classifier]

# --- 資料結構：準備存儲 RDF 計數 ---
environments_counts = 2 # 0: CNT, 1: Graphene/CNT 交界處

# 第一部分：TFSI (Polar vs Non-polar)
rdf_count_tfsi = [ [0 for _ in range(n_bins)] for _ in range(environments_counts) ]
rdf_N_tfsi = [0 for _ in range(environments_counts)]

# 第二部分：BMIM (Polar vs Non-polar)
rdf_count_bmim = [ [0 for _ in range(n_bins)] for _ in range(environments_counts) ]
rdf_N_bmim = [0 for _ in range(environments_counts)]

# --- Numba JIT 預熱 (Warm-up) ---
print("=" * 60)
print("  RDF Analysis V3 - Numba Accelerated")
print("=" * 60)
print("\n🔧 Numba JIT 編譯中 (首次執行會較慢)...")
# 預熱 Numba 函數
_dummy = compute_environment_labels(
    np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]),
    6.0, 6.0, 6.0
)
_dummy_idx0, _dummy_idx1 = classify_atoms_by_environment(np.array([2.0, 3.0]), 2)
_dummy_hist = accumulate_histogram(np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64), 10)
print("✓ Numba JIT 編譯完成！\n")

# --- 2. 開始軌跡迴圈 (Trajectory Loop) ---
print(f"📊 開始分析軌跡: frame {framestart} → {frameend}")
print(f"   總共 {frameend - framestart} 幀\n")

for t0 in tqdm(range(framestart, frameend), desc="Processing frames", unit="frame", ncols=80):
    u.trajectory[t0]
    
    # 動態生成四個我們需要的 AtomGroups
    # 中心點 (group1) 只取 z < z_max
    grp1_tfsi = u.select_atoms(f"({selections['Tf2_polar']}) and prop z < {z_max}")
    grp1_bmim = u.select_atoms(f"({selections['BMI_polar']}) and prop z < {z_max}")
    
    # 目標點 (group2) 取 z < z_max + rdf_max
    grp2_tfsi = u.select_atoms(f"({selections['Tf2_nonpolar']}) and prop z < {z_max + rdf_max}")
    grp2_bmim = u.select_atoms(f"({selections['BMI_nonpolar']}) and prop z < {z_max + rdf_max}")

    # ==========================================
    # 建立一個內部函數來處理環境分類與 RDF 計算，避免程式碼重複
    # 使用 Numba 加速環境分類
    # ==========================================
    def compute_rdf_for_center(grp_center, grp_target, rdf_count_array, rdf_N_array):
        if len(grp_center) == 0 or len(grp_target) == 0:
            return
            
        # 計算與各電極的最短距離
        contact_dist_0 = np.amin(distances.distance_array(
            grp_center.positions, electrode_groups[0].positions, 
            box=u.dimensions, backend='OpenMP'), axis=1)
        contact_dist_1 = np.amin(distances.distance_array(
            grp_center.positions, electrode_groups[1].positions, 
            box=u.dimensions, backend='OpenMP'), axis=1)
        contact_dist_2 = np.amin(distances.distance_array(
            grp_center.positions, electrode_groups[2].positions, 
            box=u.dimensions, backend='OpenMP'), axis=1)
        
        # 使用 Numba 加速環境標籤計算
        environment = compute_environment_labels(
            contact_dist_0, contact_dist_1, contact_dist_2,
            r_electrode_contact[0], r_electrode_contact[1], r_electrode_contact[2]
        )
        
        # 使用 Numba 加速原子分類
        indices_env0, indices_env1 = classify_atoms_by_environment(environment, len(grp_center))
        
        # 計算 RDF - CNT 環境 (env=2)
        if len(indices_env0) > 0:
            atoms_env0 = grp_center[indices_env0]
            pairs, dist = distances.capped_distance(
                atoms_env0.positions, grp_target.positions, 
                rdf_max, box=u.dimensions)
            hist_count = np.histogram(dist, bins=n_bins, range=(0.0, rdf_max))[0]
            rdf_count_array[0] = accumulate_histogram(
                np.array(rdf_count_array[0], dtype=np.int64), 
                hist_count.astype(np.int64), n_bins).tolist()
            rdf_N_array[0] += len(pairs)
        
        # 計算 RDF - 交界處環境 (env=3)
        if len(indices_env1) > 0:
            atoms_env1 = grp_center[indices_env1]
            pairs, dist = distances.capped_distance(
                atoms_env1.positions, grp_target.positions, 
                rdf_max, box=u.dimensions)
            hist_count = np.histogram(dist, bins=n_bins, range=(0.0, rdf_max))[0]
            rdf_count_array[1] = accumulate_histogram(
                np.array(rdf_count_array[1], dtype=np.int64), 
                hist_count.astype(np.int64), n_bins).tolist()
            rdf_N_array[1] += len(pairs)

    # --- 呼叫函數，分別計算 TFSI 與 BMIM ---
    compute_rdf_for_center(grp1_tfsi, grp2_tfsi, rdf_count_tfsi, rdf_N_tfsi)
    compute_rdf_for_center(grp1_bmim, grp2_bmim, rdf_count_bmim, rdf_N_bmim)

# --- 3. 正規化 (Normalization) 與輸出 ---
# 這裡沿用您原本計算 V (體積) 的公式
edges = np.linspace(0.0, rdf_max, n_bins + 1)
V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
volume_sphere = (4 / 3) * np.pi * rdf_max**3

# 正規化並寫入檔案邏輯... (與原版類似，為 rdf_count_tfsi 和 rdf_count_bmim 分別建立四個檔案即可)
# --- 3. 正規化 (Normalization) 與輸出 ---
print("\n⏳ 開始進行正規化與寫檔...")

# 計算每個 bin 的邊界與體積
edges = np.linspace(0.0, rdf_max, n_bins + 1)
# 計算每個 bin 的中心距離 (Bin Centers)，這將成為輸出的 X 軸
r_centers = (edges[:-1] + edges[1:]) / 2 

# 計算球殼體積與總體積
V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
volume_sphere = (4 / 3) * np.pi * rdf_max**3

# 定義一個專門處理正規化與寫入檔案的函數
def normalize_and_save(rdf_count, rdf_N, file_cnt, file_grp_t):
    # 初始化 g(r) 陣列，預設為 0
    g_r_cnt = np.zeros(n_bins)
    g_r_grp_t = np.zeros(n_bins)
    
    # 正規化 CNT 環境 (env=2，對應 index 0)
    # 物理公式: 實際計數 / 總配對數 / 該層殼的體積 * 理想狀態下的球體積
    if rdf_N[0] > 0:
        g_r_cnt = np.array(rdf_count[0]) / rdf_N[0] / V * volume_sphere
        
    # 正規化 交界處環境 (env=3，對應 index 1)
    if rdf_N[1] > 0:
        g_r_grp_t = np.array(rdf_count[1]) / rdf_N[1] / V * volume_sphere
        
    # 使用 NumPy 將 r_centers 與 g(r) 結合成兩欄並寫入硬碟
    np.savetxt(file_cnt, np.column_stack((r_centers, g_r_cnt)), 
               header="r(Angstrom) g(r)", fmt="%.6f")
    print(f"✅ 已儲存: {file_cnt} (配對數: {rdf_N[0]})")
    
    np.savetxt(file_grp_t, np.column_stack((r_centers, g_r_grp_t)), 
               header="r(Angstrom) g(r)", fmt="%.6f")
    print(f"✅ 已儲存: {file_grp_t} (配對數: {rdf_N[1]})")

# 執行 TFSI 與 BMIM 的正規化及存檔
normalize_and_save(rdf_count_tfsi, rdf_N_tfsi, 'rdf_tfsi_cnt.dat', 'rdf_tfsi_grp_t.dat')
normalize_and_save(rdf_count_bmim, rdf_N_bmim, 'rdf_bmim_cnt.dat', 'rdf_bmim_grp_t.dat')
print("=========================================\n")

### 作圖部分 (Plotting) ###

# --- 1. 設定圖表整體風格 (Plot Formatting) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# --- 2. 檔案名稱與標題設定 ---
# 定義 4 個資料檔及其對應的標題與顏色
plot_configs = [
    {
        'file': 'rdf_tfsi_cnt.dat', 
        'title': 'TFSI (Polar vs Non-polar) on CNT', 
        'color': '#1f77b4', # 藍色
        'position': 1
    },
    {
        'file': 'rdf_tfsi_grp_t.dat', 
        'title': 'TFSI (Polar vs Non-polar) at Junction', 
        'color': '#ff7f0e', # 橘色
        'position': 2
    },
    {
        'file': 'rdf_bmim_cnt.dat', 
        'title': 'BMIM (Polar vs Non-polar) on CNT', 
        'color': '#2ca02c', # 綠色
        'position': 3
    },
    {
        'file': 'rdf_bmim_grp_t.dat', 
        'title': 'BMIM (Polar vs Non-polar) at Junction', 
        'color': '#d62728', # 紅色
        'position': 4
    }
]

# --- 3. 建立 2x2 的網格圖表 (2x2 Subplots) ---
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Radial Distribution Functions at Different Interfaces', fontsize=18, fontweight='bold', y=0.95)

for config in plot_configs:
    ax = fig.add_subplot(2, 2, config['position'])
    
    file_name = config['file']
    if os.path.exists(file_name):
        # 讀取數據：第 0 欄為距離 r (Å)，第 1 欄為 g(r)
        data = np.loadtxt(file_name)
        r = data[:, 0]
        g_r = data[:, 1]
        
        # 繪製曲線
        ax.plot(r, g_r, lw=2.5, color=config['color'], label='g(r)')
        
        # 填滿曲線下方區域 (可選，增加視覺美觀)
        ax.fill_between(r, g_r, alpha=0.2, color=config['color'])
        
        # 設定軸範圍與標籤
        ax.set_xlim(0, 15) # RDF 的最大距離 rdf_max
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Distance, $r$ (Å)', fontsize=14)
        ax.set_ylabel('g(r)', fontsize=14)
        ax.set_title(config['title'], fontsize=14)
        
        # 加入網格線 (Grid lines)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 畫一條 y=1 的基準線 (Reference line for bulk density)
        ax.axhline(y=1.0, color='gray', linestyle=':', lw=2)
        
    else:
        # 如果找不到檔案，在圖表上顯示提示文字
        ax.text(0.5, 0.5, f"File Not Found:\n{file_name}", 
                ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
        ax.set_title(config['title'], fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

# --- 4. 調整版面並存檔 ---
plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # 預留上方給總標題的空間
output_image = 'rdf_4plots_combined.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"✅ 繪圖完成！圖表已儲存為：{output_image}")