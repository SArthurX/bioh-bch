# ArcFace + BioHash + BCH 人臉驗證系統流程詳解

本文檔詳細講解整個系統的運作流程，涵蓋影像處理、ArcFace 特徵提取、BioHash 原理以及 BCH 糾錯編碼。

## 3. BioHash 原理

相關文件：[test/test_biohash.cpp]

BioHash 是一種結合了生物特徵 (Fingerprint/Face) 和 令牌 (Token/Key) 的可撤銷生物特徵模板保護技術。

### 3.1 隨機投影 (Random Projection)
- **種子產生**: [get_date_seed] (或其他方式) 產生一個種子 (Seed)。這相當於用戶的 Token。
- **隨機矩陣**: [generate_random_matrix] 使用該種子生成一個偽隨機矩陣 (在這裡是 128x128)。理想情況下這應該是一個正交矩陣 (Orthonormal Projection)，以保持特徵間的距離關係，代碼中使用均勻分佈隨機數模擬。
- **投影**: [biohash_projection] 計算 $V_{bio} = Matrix \times V_{feature}$。
    - 原始特徵與隨機矩陣相乘，得到一個新的隨機化向量。
    - **目的**: 即使攻擊者截獲了 BioHash，如果不知道種子 (矩陣)，就無法還原原始特徵。

### 3.2 二值化 (Binarization)
- [binarize] 函數將投影後的連續值向量轉換為二進制向量 (0 或 1)。
- **閾值**: 選擇中位數 (median) 作為閾值。大於中位數的設為 1，小於的設為 0。
- 這產生了一個長度為 128 bits 的二進制碼。

### 3.3 可靠位元選擇 (Reliable Bit Selection)
- 並非所有生成的位元都同樣穩定。那些投影值接近閾值 (0) 的位元很容易因為噪聲而在 0 和 1 之間跳動。
- **原理**: 選擇投影值絕對值 (Magnitude) 最大的前 $K$ 個位元。這些位元離閾值最遠，最不可能出錯。
- **代碼實現**:
    - 計算每個投影值的絕對值。
    - 排序並選出前 `BIOHASH_K` (例如 64) 個索引。
    - 僅保留這些位置的位元作為最終的 BioHash 碼。

---

## 4. BCH 碼原理 (Bose-Chaudhuri-Hocquenghem Codes)

相關文件：[lib/bch/bch_codec.c] [lib/bch/bch_codec.h]

BCH 是一種強大的循環糾錯碼，用於在驗證時容忍 BioHash 的輕微變動 (噪聲)。

### 4.1 基本概念
- **伽羅瓦域 GF(2^m)**: 所有的數學運算都在有限域上進行。代碼中定義了 `bch_control` 結構來管理 GF 運算的查找表 (exp, log)。
- **參數**: 
    - `m=8`: 域的大小 (256)。
    - `n=255`: 碼字總長度 (bits)。
    - `t=23`: 糾錯能力，能糾正 23 個 bit 錯誤。

### 4.2 編碼 (Encode)
- **目標**: 為數據 (BioHash) 生成校驗位 (Parity/ECC)。
- **流程**: 
    - 輸入: 選取的 `BIOHASH_K` (64) bits。
    - [encode_bch]:通過生成多項式進行除法運算，其餘數即為 ECC。
    - 輸出: 數據 + ECC 拼接成的碼字 (Codeword)。
- **作用**: 生成的 ECC 會被存儲在模板中（Helper Data）。

### 4.3 解碼與糾錯 (Decode)
- **場景**: 當用戶驗證時，會生成一個新的 BioHash (可能與註冊時有少量位元差異)。
- **流程**:
    - [decode_bch]
    1. **伴隨式計算 (Syndrome Computation)**: 檢查數據是否滿足校驗關係。全為0則無錯。
    2. **錯誤定位多項式**: 使用 Berlekamp-Massey 算法找出錯誤位置多項式。
    3. **求根 (Root Finding)**: 代碼中使用了一種優化的算法 (BTZ/BTA) 而不是傳統的 Chien Search，來解出多項式的根，這些根對應錯誤的位置。
- **結果**: 返回錯誤的數量和位置，並修正這些位元。
- **驗證**: 如果錯誤數量小於等於 $t$ (23)，則認為驗證成功，並糾正回原始 BioHash；否則驗證失敗。

---

## 5. 程式流程總結 (Workflow)

基於 [test/test_biohash.cpp] 中的 [process_single_image]和驗證邏輯。

### 5.1 註冊流程 (Enrollment)
1. **讀取圖片**: 載入原始人臉圖。
2. **MTCNN 檢測**: 找到人臉框和5個關鍵點。
3. **ArcFace 對齊**: 根據關鍵點進行仿射變換，裁切出標準人臉。
4. **特徵提取**: 輸入 ArcFace 模型，得到 128維 歸一化特徵向量 $F$。
5. **產生種子**: 生成一個 Token (如日期種子 [seed])，並據此生成隨機矩陣 $R$。
6. **BioHash 生成**:
    - 計算 $P = R \times F$。
    - 二值化 $B = Threshold(P)$。
    - **可靠位選取**: 找出最穩定的 $K$ 個位元 $B_{reliable}$ 及其索引 $Indices$。
7. **BCH 編碼**: 
    - 對 $B_{reliable}$ 進行 BCH 編碼，生成校驗位 $ECC$。
8. **存儲模板**: 保存 `{Seed, Indices, ECC}` (注意：**不保存原始特徵，也不保存原始 BioHash**)。ECC 和 Indices 是作為 Helper Data。

### 5.2 驗證流程 (Verification)
1. **讀取圖片** (待測人臉)。
2. **預處理**: 同註冊 (MTCNN -> ArcFace -> Feature)。得到新特徵 $F'$。
3. **重現 BioHash**:
    - **讀取模板**: 獲取 `{Seed, Indices, ECC}`。
    - 使用讀取的 `Seed` 重建相同的隨機矩陣 $R$。
    - 計算 $P' = R \times F'$。
    - 二值化得到 $B'$。
4. **特徵選取**:
    - 根據模板中的 `Indices`，從 $B'$ 中提取對應的位元，得到待測 BioHash $B'_{reliable}$。
    - 這裡 $B'_{reliable}$ 可能與註冊時的 $B_{reliable}$ 有不同 (因為噪聲)。
5. **BCH 解碼嘗試**:
    - 使用 $B'_{reliable}$ 和存儲的 $ECC$ 嘗試解碼。
    - 這裡實際上是將 $B'_{reliable}$ 視為"接收到的含有錯誤的碼字的一部分"，結合 $ECC$ 去還原原始的 $B_{reliable}$。
6. **判定**:
    - 如果 BCH 解碼成功 (錯誤位元數 $\le t$)，則說明兩張人臉足夠相似，驗證**通過**。
    - 如果錯誤太多無法糾正，驗證**失敗**。

### 總結
這個流程結合了深度學習的強大識別能力 (ArcFace)、BioHash 的隱私保護特性 (可撤銷、不可逆) 以及 BCH 的容錯能力，構建了一個安全的人臉驗證系統。
