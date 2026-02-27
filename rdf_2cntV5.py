from MDAnalysis import *
import MDAnalysis.analysis.distances as distances
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm

# --- 引入分析資料 ---
topology = "start_drudes.pdb"
trajectory = "FV_NVT.dcd"
u = Universe(topology, trajectory)

# --- 參數設定區 ---
z_max = 130 # 已根據您的系統修正
rdf_max = 15
n_bins = 75
dr = float(rdf_max) / float(n_bins)
framestart = 5000
frameend = len(u.trajectory) - 1

# --- 定義 5 個目標基團 (Target Functional Groups) ---
print("🎯 正在定義陰陽離子的 5 個目標基團...")

# 陰離子 TFSI (2 個基團)
targets_tfsi = {
    'SO2_polar': "resname Tf2 and name Otf Otf1 Otf2 Otf3",  # 極性端 (氧原子)
    'CF3_nonpolar': "resname Tf2 and name Ctf Ctf1"          # 非極性端 (碳原子)
}

# 陽離子 BMIM (3 個基團)
targets_bmim = {
    'Im_polar': "resname BMI and name C1 C2 C21",            # 極性環 (咪唑環碳原子)
    'CH3_nonpolar': "resname BMI and name C3",               # 短鏈非極性端 (甲基)
    'Butyl_nonpolar': "resname BMI and name C4 C5 C51 C6"    # 長鏈非極性端 (丁基)
}

# 為了方便後續迴圈計算，我們把它們合併成一個大字典
all_targets = {**targets_tfsi, **targets_bmim}

# --- 準備存儲 RDF 計數的全新資料結構 ---
# 我們有 2 種電極中心 (CNT, Junction) 和 5 種目標基團
# 這裡使用巢狀字典 (Nested Dictionary) 來管理，讓後面的迴圈超級乾淨
rdf_results = {
    'CNT': {
        'SO2_polar': np.zeros(n_bins, dtype=np.int64),
        'CF3_nonpolar': np.zeros(n_bins, dtype=np.int64),
        'Im_polar': np.zeros(n_bins, dtype=np.int64),
        'CH3_nonpolar': np.zeros(n_bins, dtype=np.int64),
        'Butyl_nonpolar': np.zeros(n_bins, dtype=np.int64)
    },
    'Junction': {
        'SO2_polar': np.zeros(n_bins, dtype=np.int64),
        'CF3_nonpolar': np.zeros(n_bins, dtype=np.int64),
        'Im_polar': np.zeros(n_bins, dtype=np.int64),
        'CH3_nonpolar': np.zeros(n_bins, dtype=np.int64),
        'Butyl_nonpolar': np.zeros(n_bins, dtype=np.int64)
    }
}

# 電極環境定義 (Segids)
# 先抓出所有的石墨烯與 CNT
print("⏳ 正在定義並萃取電極幾何特徵 (CNT vs 交界處)...")
grp_all = u.select_atoms("segid A")
cnt_all = u.select_atoms("segid B or segid C")

# 1. 定義交界處 (Junction)：距離石墨烯 10 Å 內的 CNT 原子
center_junction = u.select_atoms("(segid B or segid C) and around 10.0 group grp", grp=grp_all)

# 2. 定義純 CNT 表面：所有的 CNT 扣除掉交界處的原子
center_cnt = cnt_all - center_junction
print(f"✅ 電極定義完成！純 CNT 原子數: {len(center_cnt)}, 交界處原子數: {len(center_junction)}")

# 3. 記錄中心原子的數量 (用於後續的正規化)
n_center_cnt = len(center_cnt)
n_center_junction = len(center_junction)

# --- 開始軌跡迴圈 (Trajectory Loop) ---
print(f"📊 開始分析軌跡: frame {framestart} → {frameend}")
print(f"   總共 {frameend - framestart} 幀\n")

# --- 進入軌跡迴圈 (Trajectory Loop) ---
print(f"📊 開始分析軌跡: frame {framestart} → {frameend}")
n_frames = frameend - framestart

for ts in tqdm(u.trajectory[framestart:frameend], desc="Processing frames", unit="frame", ncols=80):
    
    # 遍歷我們在步驟二定義的 5 個目標基團 (Target Groups)
    for target_name, sel_str in all_targets.items():
        
        # 動態選取該幀的目標原子
        # 使用 Z 軸過濾 (Z-Filtering) 大幅加速運算，避免計算遙遠的本體溶液
        target_grp = u.select_atoms(f"({sel_str}) and prop z < {z_max + rdf_max}")
        
        if len(target_grp) == 0:
            continue
            
        # ==========================================
        # 1. 計算 CNT 表面 (Environment: CNT) 的 RDF
        # ==========================================
        if n_center_cnt > 0:
            # 使用 MDAnalysis 的 capped_distance，它底層由 C 語言與 OpenMP 驅動
            # 並且能完美處理 120 度角的 六方晶系 (Hexagonal System) 週期性邊界條件 (PBC)
            pairs_cnt, dist_cnt = distances.capped_distance(
                center_cnt.positions, target_grp.positions, 
                max_cutoff=rdf_max, box=u.dimensions, backend='OpenMP'
            )
            # 統計直方圖 (Histogram) 並累加到我們的資料結構中
            counts_cnt, _ = np.histogram(dist_cnt, bins=n_bins, range=(0.0, rdf_max))
            rdf_results['CNT'][target_name] += counts_cnt

        # ==========================================
        # 2. 計算 交界處 (Environment: Junction) 的 RDF
        # ==========================================
        if n_center_junction > 0:
            pairs_junc, dist_junc = distances.capped_distance(
                center_junction.positions, target_grp.positions, 
                max_cutoff=rdf_max, box=u.dimensions, backend='OpenMP'
            )
            counts_junc, _ = np.histogram(dist_junc, bins=n_bins, range=(0.0, rdf_max))
            rdf_results['Junction'][target_name] += counts_junc

print("✅ 軌跡運算完成！")

# ==========================================
# --- 4. 正規化 (Normalization) 與資料寫入 ---
# ==========================================
print("\n⏳ 開始進行正規化與寫入 .dat 檔案...")

# 計算每個 bin 的中心距離 (r) 與對應的球殼體積 (Shell Volumes)
edges = np.linspace(0.0, rdf_max, n_bins + 1)
r_centers = (edges[:-1] + edges[1:]) / 2
shell_volumes = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
total_vol = (4/3) * np.pi * rdf_max**3

# 準備遍歷我們的巢狀字典
environments = {'CNT': n_center_cnt, 'Junction': n_center_junction}

for env_name, n_centers in environments.items():
    for target_name, counts in rdf_results[env_name].items():
        
        # 建立輸出的檔名，例如：rdf_CNT_SO2_polar.dat
        filename = f"rdf_{env_name}_{target_name}.dat"
        
        if n_centers > 0 and n_frames > 0:
            # 1. 取得「平均每一幀、每一個電極中心」在該距離區間內找到的原子數
            avg_counts = counts / (n_centers * n_frames)
            
            # 2. 計算理想氣體密度 (Ideal Gas Density) 作為收斂基準
            rho_ideal = np.sum(avg_counts) / total_vol if np.sum(avg_counts) > 0 else 1.0
            
            # 3. 計算最終的徑向分佈函數 g(r)
            g_r = avg_counts / (shell_volumes * rho_ideal) if rho_ideal > 0 else np.zeros(n_bins)
        else:
            g_r = np.zeros(n_bins)
            
        # 寫入硬碟
        np.savetxt(filename, np.column_stack([r_centers, g_r]), 
                   header="r(Angstrom) g(r)", fmt="%.6f")
        print(f"✅ 已儲存: {filename}")

print("=========================================")

# ==========================================
# --- 5. 繪製 2x2 專業比較圖 (Plotting) ---
# ==========================================
print("🎨 正在產生徑向分佈函數 (RDF) 分析圖表...")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 建立 2x2 的網格圖表，大小設定為 14x10 英吋
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Electrode-Centric Radial Distribution Functions (RDF)', fontsize=18, fontweight='bold', y=0.96)

# 定義我們要畫的線條顏色與標籤
plot_styles = {
    'SO2_polar': {'color': '#d62728', 'label': 'TFSI: $-SO_2$ (Polar)'},       # 紅色
    'CF3_nonpolar': {'color': '#1f77b4', 'label': 'TFSI: $-CF_3$ (Non-polar)'},  # 藍色
    'Im_polar': {'color': '#2ca02c', 'label': 'BMIM: $-Im$ Ring (Polar)'},       # 綠色
    'CH3_nonpolar': {'color': '#ff7f0e', 'label': 'BMIM: $-CH_3$ (Short)'},      # 橘色
    'Butyl_nonpolar': {'color': '#9467bd', 'label': 'BMIM: $-C_4H_9$ (Long)'}    # 紫色
}

# 設定 4 個子圖的配置：(環境, 離子類型, 包含的基團, 畫布位置)
subplots_config = [
    ('CNT', 'TFSI', ['SO2_polar', 'CF3_nonpolar'], axes[0, 0]),
    ('CNT', 'BMIM', ['Im_polar', 'CH3_nonpolar', 'Butyl_nonpolar'], axes[0, 1]),
    ('Junction', 'TFSI', ['SO2_polar', 'CF3_nonpolar'], axes[1, 0]),
    ('Junction', 'BMIM', ['Im_polar', 'CH3_nonpolar', 'Butyl_nonpolar'], axes[1, 1])
]

for env_name, ion_name, target_keys, ax in subplots_config:
    has_data = False
    for target in target_keys:
        file_name = f"rdf_{env_name}_{target}.dat"
        if os.path.exists(file_name):
            data = np.loadtxt(file_name)
            r, g_r = data[:, 0], data[:, 1]
            # 畫線
            ax.plot(r, g_r, lw=2.5, color=plot_styles[target]['color'], label=plot_styles[target]['label'])
            has_data = True
            
    if has_data:
        ax.axhline(y=1.0, color='gray', linestyle=':', lw=2) # 基準線
        ax.set_xlim(0, rdf_max)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Distance from Electrode, $r$ (Å)', fontsize=14)
        ax.set_ylabel('g(r)', fontsize=14)
        ax.set_title(f"{ion_name} on {env_name}", fontsize=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12, loc='upper right')
    else:
        ax.text(0.5, 0.5, "Data Not Available", ha='center', va='center', fontsize=14, color='red')

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
output_image = 'rdf_electrode_centric_results.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"🎉 恭喜！分析與繪圖完美結束。圖表已儲存為：{output_image}")