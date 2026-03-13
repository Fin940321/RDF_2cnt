# RDF計算程式碼詳細解析

## 目錄
1. [什麼是徑向分布函數 (RDF)](#什麼是徑向分布函數-rdf)
2. [物理意義與應用](#物理意義與應用)
3. [程式碼整體架構](#程式碼整體架構)
4. [詳細步驟解析](#詳細步驟解析)
5. [環境分類系統](#環境分類系統)
6. [RDF計算與正規化](#rdf計算與正規化)
7. [輸出結果說明](#輸出結果說明)

---

## 什麼是徑向分布函數 (RDF)

### 基本定義
徑向分布函數 $g(r)$ 描述了在距離某個粒子（參考粒子）距離 $r$ 處找到另一個粒子的**相對機率密度**。

數學上，RDF定義為：
$$g(r) = \frac{\rho(r)}{\rho_{bulk}}$$

其中：
- $\rho(r)$ = 在距離 $r$ 處的局部密度
- $\rho_{bulk}$ = 系統的平均密度（bulk density）

### RDF的物理意義

- **$g(r) = 1$**: 表示在距離 $r$ 處的粒子分布與隨機分布相同（無關聯）
- **$g(r) > 1$**: 表示在距離 $r$ 處找到粒子的機率**高於**隨機分布（有吸引或結構）
- **$g(r) < 1$**: 表示在距離 $r$ 處找到粒子的機率**低於**隨機分布（有排斥）
- **$g(r) = 0$**: 表示不可能在距離 $r$ 處找到粒子（例如硬球的核心排斥）

---

## 物理意義與應用

### 為什麼要計算RDF？

1. **結構分析**: 
   - 識別液體、溶液中的局部結構
   - 找出粒子間的特徵距離（如第一配位殼層）

2. **相互作用研究**:
   - 了解不同種類粒子間的親和性
   - 研究離子配對、溶劑化結構

3. **環境依賴性**:
   - 本程式碼特別研究**不同電極環境**下的RDF
   - 比較離子在CNT內部 vs. 石墨烯附近的結構差異

### 本程式碼的特殊性

這個程式碼不只計算整體的RDF，而是將離子**依照其所處的電極環境**進行分類，分別計算每個環境下的RDF。這樣可以了解：
- 電極結構如何影響離子液體的局部結構
- 不同電極環境下的離子-離子相互作用差異

---

## 程式碼整體架構

```
1. 初始化與參數設定
   ├─ 載入軌跡檔案 (PDB + DCD)
   ├─ 設定接觸距離、RDF範圍、bins數量
   └─ 定義電極選擇器（segid A-E）

2. 建立靜態電極原子群組
   └─ 因為電極固定不動，只需建立一次

3. 主迴圈：遍歷每個時間frame
   ├─ 3.1 選擇當前frame的離子
   │   ├─ Group1: TrF陰離子 (z < 55)
   │   └─ Group2: BMIM陽離子的特定原子
   │
   ├─ 3.2 計算每個離子與各電極的最短距離
   │
   ├─ 3.3 環境分類
   │   ├─ 判斷離子是否"接觸"電極（距離 < 閾值）
   │   ├─ 使用二進制編碼標記環境
   │   └─ 將離子分配到對應環境
   │
   └─ 3.4 計算RDF並累加
       ├─ TrF-BMIM RDF（陰陽離子）
       └─ TrF-TrF RDF（陰離子對）

4. 正規化與輸出
   ├─ 依照球殼體積正規化
   ├─ 依照粒子對數量正規化
   └─ 輸出4個.dat檔案
```

---

## 詳細步驟解析

### 步驟1: 初始化與參數設定

```python
topology = "start_drudes.pdb"
trajectory = "FV_NVT.dcd"
u = Universe(topology, trajectory)
```
- **作用**: 載入分子動力學模擬的結果
- **PDB檔**: 提供原子拓撲結構（誰是誰、鍵結關係）
- **DCD檔**: 提供時間序列的座標資料

```python
r_electrode_contact = np.array([ 6.0 , 6.0 , 6.0 , 6.0 ] )
```
- **物理意義**: 定義"接觸"電極的距離閾值（單位：Å）
- 如果離子與電極的距離 < 6.0 Å，則視為該離子在該電極的影響範圍內

```python
z_max = 55
rdf_max = 15
n_bins = 75
dr = float(rdf_max) / float(n_bins)  # = 0.2 Å
```
- **z_max**: 只考慮 z < 55 Å 的離子（排除過遠的區域）
- **rdf_max**: RDF計算到15 Å為止
- **n_bins**: 將0-15 Å分成75個區間
- **dr**: 每個bin的寬度 = 0.2 Å

```python
resname1='trf'   # TrF陰離子
atoms1='Cf'      # TrF的中心原子

resname2='BMI'   # BMIM陽離子
atoms2=('C1','C2','C21')  # BMIM的特定原子
```
- 定義要分析的粒子種類
- `group1` (TrF) 通常作為**參考粒子**
- `group2` (BMIM) 作為**計算距離的目標粒子**

```python
classifier=[]
classifier.append( "segid A" )  # 電極1
classifier.append( "segid B" )  # 電極2
classifier.append( "segid C" )  # 電極3
classifier.append( "segid D" )  # 電極4
classifier.append( "segid E" )  # 電極5
```
- **segid**: segment ID，MDAnalysis中用來標記不同分子群組
- 這裡定義了5個電極區域

---

### 步驟2: 建立靜態電極原子群組

```python
u.trajectory[0]  # 跳到第一個frame
electrode_groups=[]
for electrode in classifier:
    electrode_groups.append( u.select_atoms(electrode) )
```

**為什麼只在第一個frame建立？**
- 因為電極在模擬中是**固定不動**的（frozen）
- 不需要在每個frame重新選擇，節省計算時間

---

### 步驟3: 主迴圈遍歷軌跡

```python
framestart=5000
frameend=len(u.trajectory)-1

for t0 in range(framestart, frameend):
    u.trajectory[t0]
```

**跳過前5000個frames的原因**:
- 模擬剛開始時系統可能還在**平衡階段**（equilibration）
- 這段時間的數據不能代表平衡狀態，應排除

---

#### 步驟3.1: 選擇當前frame的離子

```python
group1 = u.select_atoms("name %s and resname %s and prop z < %s" 
                        % (atoms1, resname1, z_max))
```
- 選擇**所有TrF陰離子**（名稱=Cf，residue名稱=trf）
- 且 z座標 < 55 Å（排除太遠的離子）

```python
group2 = u.select_atoms("name XXX")  # 先建立空group
for ele in atoms2:
    group2 = group2 + u.select_atoms("name %s and resname %s and prop z < %s" 
                                     % (ele, resname2, z_max + rdf_max))
```
- 選擇**BMIM陽離子的C1、C2、C21原子**
- z範圍比group1大15 Å（因為RDF計算範圍是15 Å）
- 這樣確保不會漏掉可能在RDF範圍內的粒子

---

#### 步驟3.2: 計算與電極的距離

```python
contact_dist_electrodes=[]
for electrode in electrode_groups:
    dist = distances.distance_array(group1.positions, electrode.positions, 
                                    box=u.dimensions, backend='OpenMP')
    contact_dist = np.amin(dist, axis=1)
    contact_dist_electrodes.append(contact_dist)
```

**這段做了什麼？**

1. `distance_array()`: 計算group1中**每個原子**與electrode中**每個原子**的距離
   - 產生一個矩陣：`[N_group1 × N_electrode]`
   - 考慮週期性邊界條件（`box=u.dimensions`）

2. `np.amin(dist, axis=1)`: 對每個group1原子，找出與電極的**最短距離**
   - 結果是長度為 N_group1 的陣列

**物理意義**:
- `contact_dist_electrodes[i][j]` = 第j個TrF離子與第i個電極的最短距離

---

#### 步驟3.3: 環境分類（核心邏輯）

##### 3.3.1 判斷是否接觸電極

```python
close_contact_electrodes = []
close_contact_electrodes.append(
    abs(np.heaviside(contact_dist_electrodes[0] - r_electrode_contact[0], 1) - 1)
)
```

**這行數學做了什麼？**

讓我們逐步分解：

1. `contact_dist_electrodes[0] - r_electrode_contact[0]`:
   - 距離減去閾值（6.0 Å）
   - 若結果 < 0：離子靠近電極
   - 若結果 ≥ 0：離子遠離電極

2. `np.heaviside(..., 1)`:
   - Heaviside階躍函數：
     - 輸入 < 0 → 輸出 0
     - 輸入 ≥ 0 → 輸出 1
   - 所以：
     - 靠近電極 → 0
     - 遠離電極 → 1

3. `abs(... - 1)`:
   - 將0和1反轉：
     - 靠近電極 → 1
     - 遠離電極 → 0

**最終結果**: 一個0/1陣列，1表示該離子**接觸**該電極

##### 3.3.2 合併兩個CNT電極

```python
temp1 = abs(np.heaviside(contact_dist_electrodes[1] - r_electrode_contact[1], 1) - 1)
temp2 = abs(np.heaviside(contact_dist_electrodes[2] - r_electrode_contact[2], 1) - 1)
temp1 += temp2
close_contact_electrodes.append(
    np.ma.masked_not_equal(temp1, 0).mask.astype(int)
)
```

**為什麼合併？**
- 電極1和電極2是**兩個相同的CNT**
- 程式碼想把"靠近任一CNT"視為同一種環境

**數學操作**:
1. `temp1 + temp2`: 相加後可能是0、1或2
   - 0: 不接觸任何CNT
   - 1: 接觸一個CNT
   - 2: 同時接觸兩個CNT（罕見）

2. `np.ma.masked_not_equal(temp1, 0).mask.astype(int)`:
   - 製造mask：所有非0的元素 → True
   - 轉成int：True → 1, False → 0
   - **結果**: 只要接觸任一CNT就是1

##### 3.3.3 二進制環境編碼

```python
environment = np.zeros(len(close_contact_electrodes[0]))
for i in range(len(close_contact_electrodes)):
    environment += close_contact_electrodes[i] * 2**i
```

**這是核心的分類邏輯！**

使用**二進制編碼**標記環境：

- `close_contact_electrodes[0]` × $2^0$ = × 1 (電極0: 石墨烯)
- `close_contact_electrodes[1]` × $2^1$ = × 2 (電極1+2: 兩個CNT)

**可能的環境值**:
- `environment = 0` (二進制: 00): 不接觸任何電極
- `environment = 1` (二進制: 01): 只接觸石墨烯
- `environment = 2` (二進制: 10): 只接觸CNT
- `environment = 3` (二進制: 11): 同時接觸石墨烯和CNT

##### 3.3.4 分配離子到環境

```python
environment_atom = [[] for i in range(2)]
for i in range(len(group1.indices)):
    if environment[i] == 2:
        environment_atom[0].append(group1[i])  # 環境0: CNT
    elif environment[i] == 3:
        environment_atom[1].append(group1[i])  # 環境1: 石墨烯+CNT
    elif environment[i] == 0:
        pass  # 忽略不接觸的離子
```

**最終分類**:
- **環境0**: 只接觸CNT的離子
- **環境1**: 同時接觸石墨烯和CNT的離子
- **忽略**: 完全不接觸電極的離子（environment=0）、只接觸石墨烯的（environment=1）

---

#### 步驟3.4: 計算RDF

```python
for i in range(len(environment_atom)):
    if environment_atom[i]:
        # TrF-BMIM RDF
        pairs, dist = distances.capped_distance(
            AtomGroup(environment_atom[i]).positions, 
            group2.positions, 
            rdf_max, 
            box=u.dimensions
        )
        count = np.histogram(dist, bins=n_bins, range=(0.0, rdf_max))[0]
        rdf_count[i] += count
        rdf_N_N[i] += len(pairs)
```

**這段在做什麼？**

1. **`capped_distance()`**:
   - 計算兩組原子間的距離
   - **只返回距離 < rdf_max 的粒子對**（效率優化）
   - 返回：
     - `pairs`: 粒子對的索引
     - `dist`: 對應的距離

2. **`np.histogram()`**:
   - 將距離分到75個bins中
   - 例如：距離3.2 Å會落在bin 16（因為3.2/0.2=16）
   - 統計每個bin中有多少粒子對

3. **累加**:
   - `rdf_count[i] += count`: 將這個frame的統計加到累積值
   - `rdf_N_N[i] += len(pairs)`: 記錄總共有多少粒子對

**TrF-TrF RDF** (第二組):
```python
pairs, dist = distances.capped_distance(
    AtomGroup(environment_atom[i]).positions, 
    group1.positions,  # 注意：這次是group1自己
    rdf_max, 
    box=u.dimensions
)
```
- 相同邏輯，但計算TrF陰離子之間的RDF

---

### 步驟4: 正規化與輸出

#### 4.1 計算球殼體積

```python
count, edges = np.histogram(dist, bins=n_bins, range=(0.0, rdf_max))
V = (4/3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
```

**為什麼需要體積正規化？**

考慮一個球殼，內半徑 $r_1$，外半徑 $r_2$：
$$V_{shell} = \frac{4}{3}\pi (r_2^3 - r_1^3)$$

**物理意義**:
- 距離較遠的球殼體積**較大**
- 即使粒子均勻分布，遠處的球殼會統計到**更多**粒子
- 必須除以體積來消除這個效應

**例子**:
- Bin 1: 0-0.2 Å，體積很小
- Bin 75: 14.8-15.0 Å，體積很大
- 如果不正規化，遠處的bin會被高估

#### 4.2 RDF正規化

```python
volume_sphere = (4/3) * np.pi * rdf_max**3

for i in range(len(rdf_count)):
    rdf_count[i] = rdf_count[i] / rdf_N_N[i] / V * volume_sphere
```

**完整的正規化公式**:

$$g(r) = \frac{N_{pairs}(r)}{N_{total}} \times \frac{V_{sphere}}{V_{shell}(r)}$$

其中：
- $N_{pairs}(r)$ = 距離在 $r$ 附近的粒子對數量（`rdf_count[i]`）
- $N_{total}$ = 所有粒子對的總數（`rdf_N_N[i]`）
- $V_{sphere}$ = 整個RDF範圍的球體體積
- $V_{shell}(r)$ = 距離 $r$ 處的球殼體積

**物理意義的回歸**:
- 分子：實際在距離 $r$ 處的粒子密度
- 分母：如果粒子完全隨機分布的期望密度
- 比值 = $g(r)$，就是相對機率密度

#### 4.3 輸出檔案

```python
# TrF-BMIM RDF
with open("rdf_2cnt_otfbmim_cnt.dat", "w") as ofile_rdf_an0:
    for j in range(len(rdf_count[0])):
        print('{0:5.8f}  {1:5.8f}'.format(j*dr, rdf_count[0][j]), 
              file=ofile_rdf_an0)
```

**輸出4個檔案**:
1. `rdf_2cnt_otfbmim_cnt.dat`: CNT環境下的TrF-BMIM RDF
2. `rdf_2cnt_otfbmim_grp_t.dat`: 石墨烯+CNT環境下的TrF-BMIM RDF
3. `rdf_2cnt_otfotf_cnt.dat`: CNT環境下的TrF-TrF RDF
4. `rdf_2cnt_otfotf_grp_t.dat`: 石墨烯+CNT環境下的TrF-TrF RDF

**檔案格式**:
```
距離(Å)    g(r)
0.00000000  0.00000000
0.20000000  0.15234567
0.40000000  0.89012345
...
```

---

## 環境分類系統

### 視覺化說明

```
        ┌─────────────────────────────┐
        │      石墨烯電極 (segid A)    │
        └─────────────────────────────┘
                    ↓
          ╔═══════════════════╗
          ║  環境1: 同時接觸   ║
          ║  石墨烯 + CNT      ║
          ╚═══════════════════╝
                    ↓
        ┌───────────────────────────┐
        │   CNT電極 (segid B/C)     │
        │                           │
        │   ╔═══════════════════╗   │
        │   ║  環境0: 只接觸CNT ║   │
        │   ╚═══════════════════╝   │
        │                           │
        └───────────────────────────┘
```

### 環境編碼表

| environment值 | 二進制 | 石墨烯 | CNT | 分類結果 |
|--------------|--------|--------|-----|----------|
| 0            | 00     | ✗      | ✗   | 忽略     |
| 1            | 01     | ✓      | ✗   | 忽略     |
| 2            | 10     | ✗      | ✓   | 環境0    |
| 3            | 11     | ✓      | ✓   | 環境1    |

---

## RDF計算與正規化

### 為什麼要分環境計算？

**科學問題**:
- 電極的幾何結構會影響離子液體的局部結構嗎？
- CNT內部的狹窄空間會改變離子間的相互作用嗎？
- 石墨烯平面與CNT圓柱面的差異會反映在RDF上嗎？

**透過分環境的RDF可以回答**:
- 比較環境0和環境1的RDF差異
- 如果$g(r)$的峰位置或峰高不同，說明電極確實影響了離子結構

### RDF峰的解讀

假設計算出的TrF-BMIM RDF如下：

```
r (Å)    g(r)
2.0      0.0    ← 核心排斥，無法靠這麼近
4.5      2.8    ← 第一峰：最可能的陰陽離子距離
6.0      0.8    ← 谷：不太可能的距離
8.0      1.3    ← 第二峰：第二配位殼層
12.0     1.0    ← 趨近於1，接近隨機分布
```

**物理解釋**:
- **第一峰 (4.5 Å, g=2.8)**: 
  - TrF和BMIM的最穩定距離
  - g=2.8表示這個距離的機率是隨機分布的2.8倍
  - 說明有**強烈的陰陽離子配對傾向**

- **第一谷 (6.0 Å, g=0.8)**:
  - 被第一配位殼層"遮蔽"的區域
  - 相對較少粒子

- **第二峰 (8.0 Å)**:
  - 第二近鄰
  - 峰較低，說明結構相關性減弱

- **遠距離 (12.0 Å, g→1)**:
  - 失去相關性，趨近隨機分布

---

## 輸出結果說明

### 如何使用輸出檔案

#### 1. 繪製RDF圖
```python
import numpy as np
import matplotlib.pyplot as plt

# 讀取數據
data_cnt = np.loadtxt('rdf_2cnt_otfbmim_cnt.dat')
data_grp = np.loadtxt('rdf_2cnt_otfbmim_grp_t.dat')

plt.plot(data_cnt[:, 0], data_cnt[:, 1], label='CNT environment')
plt.plot(data_grp[:, 0], data_grp[:, 1], label='Graphene+CNT environment')
plt.xlabel('r (Å)')
plt.ylabel('g(r)')
plt.legend()
plt.show()
```

#### 2. 分析配位數

配位數（coordination number）可從RDF積分得到：

$$n(r) = 4\pi \rho \int_0^r g(r') r'^2 dr'$$

物理意義：在距離 $r$ 內平均有多少個鄰近粒子

```python
# 計算第一配位殼層的配位數（積分到第一谷）
rho_bulk = N_particles / Volume  # 需要知道系統密度
r_first_minimum = 6.0  # 從圖上找第一個谷的位置

# 數值積分
r = data_cnt[:, 0]
g = data_cnt[:, 1]
mask = r <= r_first_minimum
coordination = 4 * np.pi * rho_bulk * np.trapz(g[mask] * r[mask]**2, r[mask])
```

---

## 常見問題與除錯

### Q1: 為什麼RDF在r→0時是0？
**A**: 粒子不能重疊（核心排斥），在極短距離內不可能找到另一個粒子。

### Q2: 為什麼要跳過前5000個frames？
**A**: 模擬需要時間達到熱平衡。剛開始的構型可能是人為設定的，不代表真實的熱力學狀態。

### Q3: `group2`為何z範圍要比`group1`大15 Å？
**A**: 如果一個group1粒子在z=54.9 Å，它可能與z=55~70 Å的group2粒子在RDF範圍（15 Å）內。如果不包含這些粒子，會低估RDF。

### Q4: 為什麼用`capped_distance`而不是`distance_array`？
**A**: 
- `distance_array`: 計算**所有**粒子對，即使距離>15 Å也計算
- `capped_distance`: 只計算距離<15 Å的粒子對
- 對於大系統，後者快很多（只關心近距離）

### Q5: 正規化公式中的`volume_sphere`是什麼？
**A**: 
- 這是一個**人為的縮放因子**
- 確保當g(r)=1時，對應於"均勻分布在整個球體內"
- 不同程式可能用不同的約定，只要一致即可

### Q6: 如果RDF在遠處不收斂到1怎麼辦？
可能原因：
1. **RDF範圍太小**: rdf_max=15 Å可能不夠，試試增加到20-30 Å
2. **統計不足**: frames太少，增加模擬時間
3. **系統太小**: 盒子尺寸不夠大，邊界效應明顯
4. **密度不均勻**: 如果電極占據大量空間，bulk density不明確

---

## 總結

### 程式碼的核心邏輯鏈

```
軌跡檔案
   ↓
對每個frame:
   ├─ 選擇離子（group1, group2）
   ├─ 計算與電極距離
   ├─ 分類到不同環境
   └─ 統計粒子對距離直方圖
   ↓
累積所有frames的統計
   ↓
正規化（除以粒子數、除以球殼體積）
   ↓
輸出RDF數據
```

### 關鍵物理概念

1. **RDF = 結構的指紋**
   - 峰位置 → 特徵距離
   - 峰高度 → 結構強度
   - 峰寬 → 熱運動/無序程度

2. **環境分類 = 條件統計**
   - 不是計算整體RDF
   - 而是"在CNT附近的離子，它們彼此的RDF是什麼？"

3. **統計與正規化**
   - 原始數據：粒子對計數
   - 除以總粒子數：轉成機率
   - 除以體積：轉成密度
   - 除以bulk密度：轉成相對機率（RDF）

---

## 延伸思考

1. **如果要分析更多環境怎麼辦？**
   - 修改`environment_atom`陣列大小
   - 添加更多`elif`條件

2. **如果要計算其他粒子對的RDF？**
   - 修改`resname1`, `atoms1`, `resname2`, `atoms2`
   - 例如：BMIM-BMIM、Cl-BMIM等

3. **如何判斷兩個環境的RDF是否有統計顯著差異？**
   - 需要計算誤差（通常用block averaging）
   - 在論文中應提供誤差棒（error bars）

4. **為什麼選擇6.0 Å作為接觸距離？**
   - 這取決於離子、電極的尺寸
   - 可能需要從整體RDF或密度分布來決定合理的cutoff

---

**希望這份說明能幫助你理解程式碼的運作邏輯與背後的物理意義！**

如果還有疑問，建議：
1. 嘗試用小系統、少量frames運行，印出中間變數檢查
2. 繪製`contact_dist_electrodes`、`environment`的分布
3. 比較不同環境的RDF圖，看看實際差異
