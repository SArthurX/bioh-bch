#include <vector>
#include <iostream>
#include <ctime>
#include <random>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "arcface.h"
#include "mtcnn.h"

extern "C" {
#include "bch_codec.h"
}

#if NCNN_VULKAN
#include "gpu.h"
#endif

using namespace cv;
using namespace std;

// ==================== 日期種子生成 ====================
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

// ==================== BioHash 投影 ====================
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

// ==================== 二值化（中位數閾值）====================
vector<uint8_t> binarize(const vector<float>& biohash) {
    vector<float> sorted = biohash;
    sort(sorted.begin(), sorted.end());
    float median = sorted[64]; // 第 64 個元素（0-indexed）
    
    vector<uint8_t> bits(128);
    for (int i = 0; i < 128; i++) {
        bits[i] = (biohash[i] > median) ? 1 : 0;
    }
    
    return bits;
}

// ==================== BCH 編碼 ====================
vector<uint8_t> bch_encode(const vector<uint8_t>& data_bits) {
    struct bch_control* bch = init_bch(8, 4, 0); // m=8, t=4
    
    // 將位元打包成字節
    uint8_t data_bytes[16]; // 128 bits = 16 bytes
    memset(data_bytes, 0, 16);
    
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            data_bytes[i] |= (data_bits[i*8 + j] << (7-j));
        }
    }
    
    // BCH 編碼
    uint8_t ecc_bytes[4]; // 32 bits = 4 bytes
    encode_bch(bch, data_bytes, 16, ecc_bytes);
    
    // 組合數據和校驗位
    vector<uint8_t> codeword(20); // 16 + 4 = 20 bytes
    memcpy(codeword.data(), data_bytes, 16);
    memcpy(codeword.data() + 16, ecc_bytes, 4);
    
    free_bch(bch);
    return codeword;
}

// ==================== BCH 解碼與驗證 ====================
bool bch_decode_and_verify(
    const vector<uint8_t>& received_bits,
    const vector<uint8_t>& stored_codeword,
    vector<uint8_t>& corrected_bits,
    int& num_errors
) {
    struct bch_control* bch = init_bch(8, 4, 0);
    
    // 打包接收到的位元
    uint8_t recv_bytes[16];
    memset(recv_bytes, 0, 16);
    
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            recv_bytes[i] |= (received_bits[i*8 + j] << (7-j));
        }
    }
    
    // 提取存儲的校驗位
    uint8_t stored_ecc[4];
    memcpy(stored_ecc, stored_codeword.data() + 16, 4);
    
    // 計算接收數據的校驗位
    uint8_t calc_ecc[4];
    encode_bch(bch, recv_bytes, 16, calc_ecc);
    
    // 解碼
    unsigned int errloc[4]; // 最多 4 個錯誤位置
    int nerr = decode_bch(bch, recv_bytes, 16, stored_ecc, calc_ecc, 
                          nullptr, errloc);
    
    num_errors = nerr;
    
    if (nerr < 0) {
        // 錯誤數超過糾正能力
        free_bch(bch);
        return false;
    }
    
    // 糾正錯誤
    if (nerr > 0) {
        correct_bch(bch, recv_bytes, 16, errloc, nerr);
    }
    
    // 解包糾正後的位元
    corrected_bits.resize(128);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            corrected_bits[i*8 + j] = (recv_bytes[i] >> (7-j)) & 1;
        }
    }
    
    free_bch(bch);
    return true;
}

// ==================== 輔助函數：比較位元差異 ====================
vector<int> find_bit_differences(const vector<uint8_t>& bits1, const vector<uint8_t>& bits2) {
    vector<int> diff_positions;
    for (int i = 0; i < 128; i++) {
        if (bits1[i] != bits2[i]) {
            diff_positions.push_back(i);
        }
    }
    return diff_positions;
}

// ==================== 輔助函數：打印位元串 ====================
void print_bits(const vector<uint8_t>& bits, int count = 20) {
    for (int i = 0; i < count && i < bits.size(); i++) {
        cout << (int)bits[i];
    }
    if (count < bits.size()) {
        cout << "...";
    }
}

// ==================== 主程序 ====================
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

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif

    cout << "=== BioHash-BCH 人臉驗證系統 ===" << endl << endl;
    
    // 獲取日期種子
    uint32_t seed = get_date_seed();
    cout << "當前日期種子: " << seed << endl << endl;
    
    // ==================== 註冊階段 ====================
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
    
    cout << "1. ArcFace 特徵 (前10個): [";
    for (int i = 0; i < 10; i++) {
        cout << fixed << setprecision(3) << feature1[i];
        if (i < 9) cout << ", ";
    }
    cout << ", ...]" << endl;
    
    // 4. 生成隨機投影矩陣
    cout << "2. 生成隨機矩陣 (128x128) 使用種子: " << seed << endl;
    auto matrix = generate_random_matrix(seed);
    
    // 5. BioHash 投影
    vector<float> biohash1 = biohash_projection(feature1, matrix);
    cout << "3. BioHash 投影 (前10個): [";
    for (int i = 0; i < 10; i++) {
        cout << fixed << setprecision(3) << biohash1[i];
        if (i < 9) cout << ", ";
    }
    cout << ", ...]" << endl;
    
    // 6. 二值化
    vector<uint8_t> bits1 = binarize(biohash1);
    cout << "4. 二值化 (前20位): ";
    print_bits(bits1, 20);
    cout << endl;
    
    // 7. BCH 編碼
    vector<uint8_t> codeword1 = bch_encode(bits1);
    cout << "5. BCH 編碼: 160 bits (20 bytes)" << endl;
    cout << "   - 數據位: 128 bits" << endl;
    cout << "   - 校驗位: 32 bits" << endl;
    cout << "6. 存儲模板: 24 bytes (碼字 20B + 種子 4B)" << endl << endl;
    
    // ==================== 驗證階段 ====================
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
    
    cout << "1. ArcFace 特徵 (前10個): [";
    for (int i = 0; i < 10; i++) {
        cout << fixed << setprecision(3) << feature2[i];
        if (i < 9) cout << ", ";
    }
    cout << ", ...]" << endl;
    
    // 4. 使用相同種子生成矩陣
    cout << "2. 使用相同種子: " << seed << endl;
    // matrix 已經生成，直接使用
    
    // 5. BioHash 投影
    vector<float> biohash2 = biohash_projection(feature2, matrix);
    cout << "3. BioHash 投影 (前10個): [";
    for (int i = 0; i < 10; i++) {
        cout << fixed << setprecision(3) << biohash2[i];
        if (i < 9) cout << ", ";
    }
    cout << ", ...]" << endl;
    
    // 6. 二值化
    vector<uint8_t> bits2 = binarize(biohash2);
    cout << "4. 二值化 (前20位): ";
    print_bits(bits2, 20);
    cout << endl;
    
    // 找出差異
    vector<int> diff_positions = find_bit_differences(bits1, bits2);
    
    cout << endl;
    cout << "位元比較:" << endl;
    cout << "- 總位元數: 128" << endl;
    cout << "- 相同位元: " << (128 - diff_positions.size()) << endl;
    cout << "- 不同位元: " << diff_positions.size() << endl;
    
    if (!diff_positions.empty()) {
        cout << "- 錯誤位置: [";
        for (size_t i = 0; i < min(diff_positions.size(), size_t(10)); i++) {
            cout << diff_positions[i];
            if (i < min(diff_positions.size(), size_t(10)) - 1) cout << ", ";
        }
        if (diff_positions.size() > 10) cout << ", ...";
        cout << "]" << endl;
    }
    
    cout << endl;
    
    // 7. BCH 解碼與糾正
    cout << "BCH 解碼:" << endl;
    vector<uint8_t> corrected_bits;
    int num_errors;
    bool success = bch_decode_and_verify(bits2, codeword1, corrected_bits, num_errors);
    
    if (success) {
        cout << "- 檢測到 " << num_errors << " 個錯誤" << endl;
        if (num_errors > 0) {
            cout << "- 糾正中..." << endl;
            cout << "- ✓ 糾正成功！" << endl;
        } else {
            cout << "- 無錯誤，完美匹配！" << endl;
        }
        
        cout << endl;
        cout << "糾正後位元 (前20位): ";
        print_bits(corrected_bits, 20);
        cout << endl;
        cout << "註冊時位元 (前20位): ";
        print_bits(bits1, 20);
        cout << endl;
        
        // 驗證是否完全匹配
        bool perfect_match = true;
        for (int i = 0; i < 128; i++) {
            if (corrected_bits[i] != bits1[i]) {
                perfect_match = false;
                break;
            }
        }
        
        if (perfect_match) {
            cout << "                      ✓ 完全匹配" << endl;
        }
        
        cout << endl;
        cout << "結果: ✓✓✓ 驗證通過 - 是同一人 ✓✓✓" << endl;
    } else {
        cout << "- 錯誤數超過糾正能力 (> 4 bits)" << endl;
        cout << "- ✗ 糾正失敗" << endl;
        cout << endl;
        cout << "結果: ✗✗✗ 驗證失敗 - 不是同一人 ✗✗✗" << endl;
    }
    
    cout << endl;
    
    // ==================== 統計信息 ====================
    cout << "=== 統計信息 ===" << endl;
    
    // 計算餘弦相似度
    float cosine_sim = calcSimilar(feature1, feature2);
    cout << "- ArcFace 餘弦相似度: " << fixed << setprecision(4) << cosine_sim << endl;
    cout << "- BioHash 位元差異: " << diff_positions.size() << "/128 (" 
         << fixed << setprecision(1) << (diff_positions.size() * 100.0 / 128.0) << "%)" << endl;
    cout << "- BCH 糾正能力: 4 bits" << endl;
    cout << "- 存儲需求: 24 bytes" << endl;

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif

    return 0;
}
