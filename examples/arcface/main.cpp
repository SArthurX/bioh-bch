#include <vector>
#include <iostream>
#include <ctime>
#include <random>
#include <algorithm>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "arcface.h"
#include "mtcnn.h"

extern "C" {
#include "bch_codec.h"
}

using namespace cv;
using namespace std;

// ==================== BCH 參數 ====================
// BCH(n, k, t) 參數說明：
// - n: 碼字長度 = 2^m - 1
// - k: 資料位元數（我們要保護的位元數）
// - t: 糾錯能力（可糾正的錯誤位元數）
// - m: GF(2^m) 的參數

#define BCH_M           8       // GF(2^8)，碼字長度 n = 255
#define BCH_T           23      // 糾錯能力：可糾正 23 個錯誤
#define BIOHASH_K       64      // 資料位元數：選擇最穩定的 64 個位元
#define BIOHASH_K_BYTES ((BIOHASH_K + 7) / 8)  // 資料位元組數 = 8 bytes
#define BIOHASH_TOTAL   128     // 原始 BioHash 投影維度

uint32_t get_date_seed() {
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    
    uint32_t year = timeinfo->tm_year + 1900;
    uint32_t month = timeinfo->tm_mon + 1;
    uint32_t day = timeinfo->tm_mday;
    
    uint32_t seed = year * 10000 + month * 100 + day;
    return seed;
}

// ==================== 隨機投影矩陣生成 ====================
vector<vector<float>> generate_random_matrix(uint32_t seed) {
    mt19937 rng(seed);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    vector<vector<float>> matrix(128, vector<float>(128));
    
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            matrix[i][j] = dist(rng);
        }
    }
    
    return matrix;
}

// =============== BioHash 投影 =================
vector<float> biohash_projection(
    const vector<float>& feature,
    const vector<vector<float>>& matrix
) {
    vector<float> projected(128, 0.0f);
    
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            projected[i] += matrix[i][j] * feature[j];
        }
    }
    
    return projected;
}

// ================ 二值化 ================
vector<uint8_t> binarize(const vector<float>& biohash) {
    vector<float> sorted = biohash;
    sort(sorted.begin(), sorted.end());
    float median = sorted[BIOHASH_TOTAL / 2];
    
    vector<uint8_t> bits(BIOHASH_TOTAL);
    for (int i = 0; i < BIOHASH_TOTAL; i++) {
        bits[i] = (biohash[i] > median) ? 1 : 0;
    }
    
    return bits;
}

// ==================== Reliable Bit Selection ====================
// 選擇投影值絕對值最大的 K 個位元（最穏定的位元）
// 返回：selected_bits（選中的位元值）和 selected_indices（選中的位置索引）
void select_reliable_bits(
    const vector<float>& biohash,
    const vector<uint8_t>& all_bits,
    vector<uint8_t>& selected_bits,
    vector<int>& selected_indices
) {
    vector<pair<int, float>> ranked(BIOHASH_TOTAL);
    for (int i = 0; i < BIOHASH_TOTAL; i++) {
        ranked[i] = {i, abs(biohash[i])};
    }
    
    sort(ranked.begin(), ranked.end(), 
         [](const pair<int, float>& a, const pair<int, float>& b) {
             return a.second > b.second;
         });
    
    selected_bits.resize(BIOHASH_K);
    selected_indices.resize(BIOHASH_K);
    
    for (int i = 0; i < BIOHASH_K; i++) {
        int idx = ranked[i].first;
        selected_indices[i] = idx;
        selected_bits[i] = all_bits[idx];
    }
    
    vector<pair<int, uint8_t>> index_bit_pairs(BIOHASH_K);
    for (int i = 0; i < BIOHASH_K; i++) {
        index_bit_pairs[i] = {selected_indices[i], selected_bits[i]};
    }
    sort(index_bit_pairs.begin(), index_bit_pairs.end());
    
    for (int i = 0; i < BIOHASH_K; i++) {
        selected_indices[i] = index_bit_pairs[i].first;
        selected_bits[i] = index_bit_pairs[i].second;
    }
}

// ==================== BCH encode ====================
// 返回：碼字（數據 + ECC）以及 ECC 長度
vector<uint8_t> bch_encode(const vector<uint8_t>& data_bits, int& ecc_bytes_len) {
    struct bch_control* bch = init_bch(BCH_M, BCH_T, 0);
    if (!bch) {
        cerr << "Error: Failed to initialize BCH with m=" << BCH_M << ", t=" << BCH_T << endl;
        ecc_bytes_len = 0;
        return {};
    }
    
    ecc_bytes_len = bch->ecc_bytes;
    
    uint8_t data_bytes[BIOHASH_K_BYTES];
    memset(data_bytes, 0, BIOHASH_K_BYTES);
    
    for (int i = 0; i < BIOHASH_K_BYTES; i++) {
        for (int j = 0; j < 8 && (i*8 + j) < BIOHASH_K; j++) {
            data_bytes[i] |= (data_bits[i*8 + j] << (7-j));
        }
    }
    
    // BCH encode
    vector<uint8_t> ecc_bytes(ecc_bytes_len);
    encode_bch(bch, data_bytes, BIOHASH_K_BYTES, ecc_bytes.data());
    
    // 組合數據和校驗位
    vector<uint8_t> codeword(BIOHASH_K_BYTES + ecc_bytes_len);
    memcpy(codeword.data(), data_bytes, BIOHASH_K_BYTES);
    memcpy(codeword.data() + BIOHASH_K_BYTES, ecc_bytes.data(), ecc_bytes_len);
    
    free_bch(bch);
    return codeword;
}

bool bch_decode_and_verify(
    const vector<uint8_t>& received_bits,
    const vector<uint8_t>& stored_codeword,
    int ecc_bytes_len,
    vector<uint8_t>& corrected_bits,
    int& num_errors
) {
    struct bch_control* bch = init_bch(BCH_M, BCH_T, 0);
    if (!bch) {
        cerr << "Error: Failed to initialize BCH" << endl;
        return false;
    }
    
    uint8_t recv_bytes[BIOHASH_K_BYTES];
    memset(recv_bytes, 0, BIOHASH_K_BYTES);
    
    for (int i = 0; i < BIOHASH_K_BYTES; i++) {
        for (int j = 0; j < 8 && (i*8 + j) < BIOHASH_K; j++) {
            recv_bytes[i] |= (received_bits[i*8 + j] << (7-j));
        }
    }
    
    // 提取存儲的校驗位
    vector<uint8_t> stored_ecc(ecc_bytes_len);
    memcpy(stored_ecc.data(), stored_codeword.data() + BIOHASH_K_BYTES, ecc_bytes_len);
    
    // 計算接收數據的校驗位
    vector<uint8_t> calc_ecc(ecc_bytes_len);
    encode_bch(bch, recv_bytes, BIOHASH_K_BYTES, calc_ecc.data());
    
    // decode
    vector<unsigned int> errloc(BCH_T);
    int nerr = decode_bch(bch, recv_bytes, BIOHASH_K_BYTES, stored_ecc.data(), calc_ecc.data(), 
                          nullptr, errloc.data());
    
    num_errors = nerr;
    
    if (nerr < 0) {
        // 錯誤數超過糾正能力
        free_bch(bch);
        return false;
    }
    
    // 糾正錯誤
    if (nerr > 0) {
        correct_bch(bch, recv_bytes, BIOHASH_K_BYTES, errloc.data(), nerr);
    }
    
    // 解包糾正後的位元
    corrected_bits.resize(BIOHASH_K);
    for (int i = 0; i < BIOHASH_K_BYTES; i++) {
        for (int j = 0; j < 8 && (i*8 + j) < BIOHASH_K; j++) {
            corrected_bits[i*8 + j] = (recv_bytes[i] >> (7-j)) & 1;
        }
    }
    
    free_bch(bch);
    return true;
}

vector<int> find_bit_differences(const vector<uint8_t>& bits1, const vector<uint8_t>& bits2) {
    vector<int> diff_positions;
    for (int i = 0; i < 128; i++) {
        if (bits1[i] != bits2[i]) {
            diff_positions.push_back(i);
        }
    }
    return diff_positions;
}

void print_bits(const vector<uint8_t>& bits, int count = 20) {
    for (int i = 0; i < count && i < bits.size(); i++) {
        cout << (int)bits[i];
    }
    if (count < bits.size()) {
        cout << "...";
    }
}

int main(int argc, char* argv[])
{
    Mat img1, img2;
    
    if (argc == 3) {
        img1 = imread(argv[1]);
        img2 = imread(argv[2]);
    } else {
        img1 = imread("../image/image.png");
        img2 = imread("../image/image2.png");
    }
    
    if (img1.empty() || img2.empty()) {
        cerr << "Error: Cannot load images!" << endl;
        return -1;
    }
    
    ncnn::Mat ncnn_img1 = ncnn::Mat::from_pixels(img1.data, ncnn::Mat::PIXEL_BGR, img1.cols, img1.rows);
    ncnn::Mat ncnn_img2 = ncnn::Mat::from_pixels(img2.data, ncnn::Mat::PIXEL_BGR, img2.cols, img2.rows);

    cout << "=== bioh-bch ===" << endl << endl;
    
    cout << "BCH 配置: BCH(255, " << BIOHASH_K << ", t=" << BCH_T << ")" << endl;
    cout << "- 資料位元: " << BIOHASH_K << " bits (選擇最穩定的位元)" << endl;
    cout << "- 糾錯能力: " << BCH_T << " bits" << endl << endl;
    
    uint32_t seed = get_date_seed();
    cout << "當前日期種子: " << seed << endl << endl;
    
    cout << "=== 註冊階段 ===" << endl;
    cout << "圖片 1: " << (argc >= 2 ? argv[1] : "../image/image.png") << endl;
    
    // 1. 人臉檢測
    MtcnnDetector detector("../models");
    vector<FaceInfo> results1 = detector.Detect(ncnn_img1);
    
    if (results1.empty()) {
        cerr << "Error: No face detected in image 1!" << endl;
        return -1;
    }
    
    // 2. 人臉對齊
    ncnn::Mat det1 = preprocess(ncnn_img1, results1[0]);
    
    // 3. ArcFace 特徵提取
    Arcface arc("../models");
    vector<float> feature1 = arc.getFeature(det1);
    
    cout << "1. ArcFace 特徵提取完成 (128維)" << endl;
    
    // 4. 生成隨機投影矩陣
    cout << "2. 生成隨機矩陣 (128x128) 使用種子: " << seed << endl;
    auto matrix = generate_random_matrix(seed);
    
    // 5. BioHash 投影
    vector<float> biohash1 = biohash_projection(feature1, matrix);
    cout << "3. BioHash 投影完成 (128維)" << endl;
    
    // 6. 二值化（全部 128 位）
    vector<uint8_t> all_bits1 = binarize(biohash1);
    cout << "4. 二值化完成 (" << BIOHASH_TOTAL << " bits)" << endl;
    
    // 7. Reliable Bit Selection - 選擇最穩定的 K 個位元
    vector<uint8_t> selected_bits1;
    vector<int> selected_indices1;
    select_reliable_bits(biohash1, all_bits1, selected_bits1, selected_indices1);
    cout << "5. Reliable Bit Selection: 選擇最穩定的 " << BIOHASH_K << " 個位元" << endl;
    cout << "   選擇的位置 (前10個): [";
    for (int i = 0; i < min(10, BIOHASH_K); i++) {
        cout << selected_indices1[i];
        if (i < 9) cout << ", ";
    }
    cout << ", ...]" << endl;
    
    // 8. BCH 編碼
    int ecc_bytes_len = 0;
    vector<uint8_t> codeword1 = bch_encode(selected_bits1, ecc_bytes_len);
    
    if (codeword1.empty()) {
        cerr << "Error: BCH encoding failed!" << endl;
        return -1;
    }
    
    int total_bytes = BIOHASH_K_BYTES + ecc_bytes_len;
    cout << "6. BCH 編碼完成" << endl;
    cout << "   - 數據位: " << BIOHASH_K << " bits (" << BIOHASH_K_BYTES << " bytes)" << endl;
    cout << "   - 校驗位: " << (ecc_bytes_len * 8) << " bits (" << ecc_bytes_len << " bytes)" << endl;
    cout << "7. 存儲模板: " << (total_bytes + 4 + BIOHASH_K * sizeof(int)) 
         << " bytes (碼字 " << total_bytes << "B + 種子 4B + 索引 " << BIOHASH_K * sizeof(int) << "B)" << endl << endl;
    
    cout << "=== 驗證階段 ===" << endl;
    cout << "圖片 2: " << (argc >= 3 ? argv[2] : "../image/image2.png") << " (同一人)" << endl;
    
    // 1. 人臉檢測
    vector<FaceInfo> results2 = detector.Detect(ncnn_img2);
    
    if (results2.empty()) {
        cerr << "Error: No face detected in image 2!" << endl;
        return -1;
    }
    
    // 2. 人臉對齊
    ncnn::Mat det2 = preprocess(ncnn_img2, results2[0]);
    
    // 3. ArcFace 特徵提取
    vector<float> feature2 = arc.getFeature(det2);
    cout << "1. ArcFace 特徵提取完成" << endl;
    
    // 4. 使用相同種子生成矩陣
    cout << "2. 使用相同種子: " << seed << endl;
    // matrix 已經生成，直接使用
    
    // 5. BioHash 投影
    vector<float> biohash2 = biohash_projection(feature2, matrix);
    cout << "3. BioHash 投影完成" << endl;
    
    // 6. 二值化
    vector<uint8_t> all_bits2 = binarize(biohash2);
    cout << "4. 二值化完成" << endl;
    
    // 7. 使用相同的索引位置提取位元（驗證時使用註冊時存儲的索引）
    vector<uint8_t> selected_bits2(BIOHASH_K);
    for (int i = 0; i < BIOHASH_K; i++) {
        selected_bits2[i] = all_bits2[selected_indices1[i]];
    }
    cout << "5. 使用存儲的索引提取對應位元" << endl;
    
    // 比較選定的位元差異
    vector<int> diff_positions;
    for (int i = 0; i < BIOHASH_K; i++) {
        if (selected_bits1[i] != selected_bits2[i]) {
            diff_positions.push_back(i);
        }
    }
    
    cout << endl;
    cout << "位元比較 (Reliable Bits):" << endl;
    cout << "- 總位元數: " << BIOHASH_K << endl;
    cout << "- 相同位元: " << (BIOHASH_K - diff_positions.size()) << endl;
    cout << "- 不同位元: " << diff_positions.size() << endl;
    
    if (!diff_positions.empty()) {
        cout << "- 錯誤位置: [";
        for (size_t i = 0; i < min(diff_positions.size(), size_t(10)); i++) {
            cout << diff_positions[i];
            if (i < min(diff_positions.size(), size_t(10)) - 1) cout << ", ";
        }
        if (diff_positions.size() > 10) cout << ", ...";
        cout << "]" << endl;
        
        cout << "錯誤位置分析 (投影值絕對值):" << endl;
        for (size_t i = 0; i < min(diff_positions.size(), size_t(10)); i++) {
            int bit_idx = diff_positions[i];
            int orig_idx = selected_indices1[bit_idx];
            float val = biohash1[orig_idx];
            cout << "  位置 " << bit_idx << " (原始索引 " << orig_idx << "): |" 
                 << fixed << setprecision(3) << abs(val) << "|" << endl;
        }
    }
        
    cout << endl;
    
    // 8. BCH 解碼與糾正
    cout << "BCH 解碼:" << endl;
    vector<uint8_t> corrected_bits;
    int num_errors;
    bool success = bch_decode_and_verify(selected_bits2, codeword1, ecc_bytes_len, corrected_bits, num_errors);
    
    if (success) {
        cout << "- 檢測到 " << num_errors << " 個錯誤" << endl;
        if (num_errors > 0) {
            cout << "- 糾正中..." << endl;
            cout << "- ✓ 糾正成功！" << endl;
        } else {
            cout << "- 無錯誤 -" << endl;
        }
        
        cout << endl;
        cout << "糾正後位元 (前20位): ";
        print_bits(corrected_bits, 20);
        cout << endl;
        cout << "註冊時位元 (前20位): ";
        print_bits(selected_bits1, 20);
        cout << endl;
        
        bool perfect_match = true;
        for (int i = 0; i < BIOHASH_K; i++) {
            if (corrected_bits[i] != selected_bits1[i]) {
                perfect_match = false;
                break;
            }
        }
        
        if (perfect_match) {
            cout << "                      ✓ 完全匹配" << endl;
        }
        
        cout << endl;
        cout << "結果: ✓ 驗證通過 - 是同一人" << endl;
    } else {
        cout << "- 錯誤數超過糾正能力 (> " << BCH_T << " bits)" << endl;
        cout << "- ✗ 糾正失敗" << endl;
        cout << endl;
        cout << "結果: ✗ 驗證失敗 - 不是同一人" << endl;
    }
    

    cout << "\n==========================================\n";
    
    float cosine_sim = calcSimilar(feature1, feature2);
    string same[] = {" 同個人", " 不是同一個人"};
    cout << "- ArcFace 餘弦相似度: " << fixed << setprecision(4) << cosine_sim << same[cosine_sim < 0.5] << endl;
    cout << "- BioHash 位元差異 (穩定位元): " << diff_positions.size() << "/" << BIOHASH_K << " (" 
         << fixed << setprecision(1) << (diff_positions.size() * 100.0 / BIOHASH_K) << "%)" << endl;
    cout << "- BCH 糾正能力: " << BCH_T << " bits" << endl;
    cout << "- 存儲需求: " << (total_bytes + 4 + BIOHASH_K * sizeof(int)) << " bytes" << endl;

    return 0;
}
