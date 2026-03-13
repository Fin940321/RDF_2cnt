from MDAnalysis import *
import MDAnalysis.analysis.distances as distances
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

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

# --- 2. 開始軌跡迴圈 (Trajectory Loop) ---
for t0 in range(framestart, frameend):
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
    # ==========================================
    def compute_rdf_for_center(grp_center, grp_target, rdf_count_array, rdf_N_array):
        if len(grp_center) == 0 or len(grp_target) == 0:
            return
            
        contact_dist_electrodes = []
        for electrode in electrode_groups:
            dist = distances.distance_array(grp_center.positions, electrode.positions, box=u.dimensions, backend='OpenMP')
            contact_dist_electrodes.append(np.amin(dist, axis=1))
            
        close_contact_electrodes = []
        # 1st electrode (Graphene)
        close_contact_electrodes.append(abs(np.heaviside(contact_dist_electrodes[0] - r_electrode_contact[0], 1) - 1))
        # 2nd & 3rd electrodes (CNTs) combined
        temp1 = abs(np.heaviside(contact_dist_electrodes[1] - r_electrode_contact[1], 1) - 1)
        temp2 = abs(np.heaviside(contact_dist_electrodes[2] - r_electrode_contact[2], 1) - 1)
        close_contact_electrodes.append(np.ma.masked_not_equal(temp1 + temp2, 0).mask.astype(int))
        
        # 計算環境標籤 (Environment Labels)
        environment = np.zeros(len(close_contact_electrodes[0]))
        for i in range(len(close_contact_electrodes)):
            environment += close_contact_electrodes[i] * (2**i)
            
        # 將中心原子分發到對應環境
        env_atoms = {0: [], 1: []} # 0對應 cnt(env=2), 1對應 交界處(env=3)
        for i in range(len(grp_center.indices)):
            if environment[i] == 2:
                env_atoms[0].append(grp_center[i])
            elif environment[i] == 3:
                env_atoms[1].append(grp_center[i])
                
        # 計算 RDF
        for env_idx, atoms_list in env_atoms.items():
            if atoms_list:
                pairs, dist = distances.capped_distance(AtomGroup(atoms_list).positions, grp_target.positions, rdf_max, box=u.dimensions)
                count = np.histogram(dist, bins=n_bins, range=(0.0, rdf_max))[0]
                rdf_count_array[env_idx] += count
                rdf_N_array[env_idx] += len(pairs)

    # --- 呼叫函數，分別計算 TFSI 與 BMIM ---
    compute_rdf_for_center(grp1_tfsi, grp2_tfsi, rdf_count_tfsi, rdf_N_tfsi)
    compute_rdf_for_center(grp1_bmim, grp2_bmim, rdf_count_bmim, rdf_N_bmim)

# --- 3. 正規化 (Normalization) 與輸出 ---
# 這裡沿用您原本計算 V (體積) 的公式
edges = np.linspace(0.0, rdf_max, n_bins + 1)
V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
volume_sphere = (4 / 3) * np.pi * rdf_max**3

# 正規化並寫入檔案邏輯... (與原版類似，為 rdf_count_tfsi 和 rdf_count_bmim 分別建立四個檔案即可)

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