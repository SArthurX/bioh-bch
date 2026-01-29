#include <vector>
#include <iostream>
#include <ctime>
#include <random>
#include <algorithm>
#include <string>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "arcface.h"
#include "mtcnn.h"

extern "C" {
#include "bch_codec.h"
}

using namespace cv;
using namespace std;

// ==================== BCH 參數 ====================
#define BCH_M           8       // GF(2^8)，碼字長度 n = 255
#define BCH_T           23      // 糾錯能力：可糾正 23 個錯誤
#define BIOHASH_K       64      // 資料位元數：選擇最穩定的 64 個位元
#define BIOHASH_K_BYTES ((BIOHASH_K + 7) / 8)  // 資料位元組數 = 8 bytes
#define BIOHASH_TOTAL   128     // 原始 BioHash 投影維度

// ==================== 輔助函數 ====================
void print_line(const string& title) {
    cout << "\n========== " << title << " ==========" << endl;
}

void print_step(int step, const string& desc) {
    cout << "[Step " << step << "] " << desc << endl;
}

string bytes_to_hex(const vector<uint8_t>& bytes) {
    stringstream ss;
    for (uint8_t b : bytes) {
        ss << hex << setfill('0') << setw(2) << (int)b;
    }
    return ss.str();
}

vector<uint8_t> hex_to_bytes(const string& hex) {
    vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        string byteStr = hex.substr(i, 2);
        uint8_t byte = (uint8_t)strtol(byteStr.c_str(), nullptr, 16);
        bytes.push_back(byte);
    }
    return bytes;
}

void print_bits(const vector<uint8_t>& bits) {
    for (size_t i = 0; i < bits.size(); i++) {
        cout << (int)bits[i];
        if ((i + 1) % 8 == 0) cout << " ";
    }
    cout << endl;
}

void print_float_array(const vector<float>& arr) {
    cout << "[";
    for (size_t i = 0; i < arr.size(); i++) {
        cout << fixed << setprecision(4) << arr[i];
        if (i < arr.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

// ==================== 核心函數 ====================
uint32_t get_date_seed() {
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    uint32_t year = timeinfo->tm_year + 1900;
    uint32_t month = timeinfo->tm_mon + 1;
    uint32_t day = timeinfo->tm_mday;
    return year * 10000 + month * 100 + day;
}

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

vector<float> biohash_projection(const vector<float>& feature, const vector<vector<float>>& matrix) {
    vector<float> projected(128, 0.0f);
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            projected[i] += matrix[i][j] * feature[j];
        }
    }
    return projected;
}

vector<uint8_t> binarize(const vector<float>& biohash, float& median_out) {
    vector<float> sorted = biohash;
    sort(sorted.begin(), sorted.end());
    median_out = sorted[BIOHASH_TOTAL / 2];
    
    vector<uint8_t> bits(BIOHASH_TOTAL);
    for (int i = 0; i < BIOHASH_TOTAL; i++) {
        bits[i] = (biohash[i] > median_out) ? 1 : 0;
    }
    return bits;
}

void select_reliable_bits(
    const vector<float>& biohash,
    const vector<uint8_t>& all_bits,
    vector<uint8_t>& selected_bits,
    vector<int>& selected_indices,
    vector<float>& selected_magnitudes
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
    selected_magnitudes.resize(BIOHASH_K);
    
    for (int i = 0; i < BIOHASH_K; i++) {
        int idx = ranked[i].first;
        selected_indices[i] = idx;
        selected_bits[i] = all_bits[idx];
        selected_magnitudes[i] = ranked[i].second;
    }
    
    // 按索引排序以保持一致的順序
    vector<tuple<int, uint8_t, float>> index_bit_pairs(BIOHASH_K);
    for (int i = 0; i < BIOHASH_K; i++) {
        index_bit_pairs[i] = {selected_indices[i], selected_bits[i], selected_magnitudes[i]};
    }
    sort(index_bit_pairs.begin(), index_bit_pairs.end());
    
    for (int i = 0; i < BIOHASH_K; i++) {
        selected_indices[i] = get<0>(index_bit_pairs[i]);
        selected_bits[i] = get<1>(index_bit_pairs[i]);
        selected_magnitudes[i] = get<2>(index_bit_pairs[i]);
    }
}

vector<uint8_t> bch_encode(const vector<uint8_t>& data_bits, int& ecc_bytes_len) {
    struct bch_control* bch = init_bch(BCH_M, BCH_T, 0);
    if (!bch) {
        cerr << "Error: Failed to initialize BCH" << endl;
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
    
    vector<uint8_t> ecc_bytes(ecc_bytes_len);
    encode_bch(bch, data_bytes, BIOHASH_K_BYTES, ecc_bytes.data());
    
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
    if (!bch) return false;
    
    uint8_t recv_bytes[BIOHASH_K_BYTES];
    memset(recv_bytes, 0, BIOHASH_K_BYTES);
    
    for (int i = 0; i < BIOHASH_K_BYTES; i++) {
        for (int j = 0; j < 8 && (i*8 + j) < BIOHASH_K; j++) {
            recv_bytes[i] |= (received_bits[i*8 + j] << (7-j));
        }
    }
    
    vector<uint8_t> stored_ecc(ecc_bytes_len);
    memcpy(stored_ecc.data(), stored_codeword.data() + BIOHASH_K_BYTES, ecc_bytes_len);
    
    vector<uint8_t> calc_ecc(ecc_bytes_len);
    encode_bch(bch, recv_bytes, BIOHASH_K_BYTES, calc_ecc.data());
    
    vector<unsigned int> errloc(BCH_T);
    int nerr = decode_bch(bch, recv_bytes, BIOHASH_K_BYTES, stored_ecc.data(), calc_ecc.data(), 
                          nullptr, errloc.data());
    
    num_errors = nerr;
    
    if (nerr < 0) {
        free_bch(bch);
        return false;
    }
    
    if (nerr > 0) {
        correct_bch(bch, recv_bytes, BIOHASH_K_BYTES, errloc.data(), nerr);
    }
    
    corrected_bits.resize(BIOHASH_K);
    for (int i = 0; i < BIOHASH_K_BYTES; i++) {
        for (int j = 0; j < 8 && (i*8 + j) < BIOHASH_K; j++) {
            corrected_bits[i*8 + j] = (recv_bytes[i] >> (7-j)) & 1;
        }
    }
    
    free_bch(bch);
    return true;
}

// ==================== 模板結構 ====================
struct BioHashTemplate {
    uint32_t seed;
    vector<int> indices;      // BIOHASH_K 個索引
    vector<uint8_t> codeword; // BCH 碼字
    int ecc_bytes_len;
    
    string to_hex() const {
        vector<uint8_t> data;
        // 4 bytes seed
        data.push_back((seed >> 24) & 0xFF);
        data.push_back((seed >> 16) & 0xFF);
        data.push_back((seed >> 8) & 0xFF);
        data.push_back(seed & 0xFF);
        // 1 byte ecc_len
        data.push_back((uint8_t)ecc_bytes_len);
        // indices (each as 1 byte since max is 127)
        for (int idx : indices) {
            data.push_back((uint8_t)idx);
        }
        // codeword
        for (uint8_t b : codeword) {
            data.push_back(b);
        }
        return bytes_to_hex(data);
    }
    
    static BioHashTemplate from_hex(const string& hex) {
        BioHashTemplate t;
        vector<uint8_t> data = hex_to_bytes(hex);
        size_t pos = 0;
        
        // seed
        t.seed = ((uint32_t)data[pos] << 24) | ((uint32_t)data[pos+1] << 16) |
                 ((uint32_t)data[pos+2] << 8) | data[pos+3];
        pos += 4;
        
        // ecc_len
        t.ecc_bytes_len = data[pos++];
        
        // indices
        t.indices.resize(BIOHASH_K);
        for (int i = 0; i < BIOHASH_K; i++) {
            t.indices[i] = data[pos++];
        }
        
        // codeword
        int codeword_len = BIOHASH_K_BYTES + t.ecc_bytes_len;
        t.codeword.resize(codeword_len);
        for (int i = 0; i < codeword_len; i++) {
            t.codeword[i] = data[pos++];
        }
        
        return t;
    }
};

// ==================== 處理單張圖片（註冊流程）====================
BioHashTemplate process_single_image(const string& image_path, bool verbose = true) {
    BioHashTemplate tmpl;
    
    if (verbose) print_line("載入圖片");
    Mat img = imread(image_path);
    if (img.empty()) {
        cerr << "Error: Cannot load image: " << image_path << endl;
        return tmpl;
    }
    if (verbose) {
        cout << "圖片路徑: " << image_path << endl;
        cout << "圖片尺寸: " << img.cols << "x" << img.rows << endl;
    }
    
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    
    // Step 1: 人臉檢測
    if (verbose) print_line("Step 1: 人臉檢測 (MTCNN)");
    MtcnnDetector detector("../models");
    vector<FaceInfo> results = detector.Detect(ncnn_img);
    
    if (results.empty()) {
        cerr << "Error: No face detected!" << endl;
        return tmpl;
    }
    
    if (verbose) {
        cout << "檢測到人臉數: " << results.size() << endl;
        cout << "人臉框: [x=" << results[0].x[0] << ", y=" << results[0].y[0] << "]" << endl;
        cout << "人臉信心度: " << results[0].score << endl;
    }
    
    // Step 2: 人臉對齊
    if (verbose) print_line("Step 2: 人臉對齊");
    ncnn::Mat aligned = preprocess(ncnn_img, results[0]);
    if (verbose) {
        cout << "對齊後尺寸: " << aligned.w << "x" << aligned.h << endl;
    }
    
    // Step 3: ArcFace 特徵提取
    if (verbose) print_line("Step 3: ArcFace 特徵提取");
    Arcface arc("../models");
    vector<float> feature = arc.getFeature(aligned);
    if (verbose) {
        cout << "特徵維度: " << feature.size() << endl;
        cout << "特徵向量: ";
        print_float_array(feature);
        
        // 計算特徵向量的一些統計量
        float min_val = *min_element(feature.begin(), feature.end());
        float max_val = *max_element(feature.begin(), feature.end());
        float sum = 0;
        for (float f : feature) sum += f;
        cout << "特徵統計: min=" << fixed << setprecision(4) << min_val 
             << ", max=" << max_val << ", mean=" << (sum/feature.size()) << endl;
    }
    
    // Step 4: 生成隨機矩陣
    if (verbose) print_line("Step 4: 隨機投影矩陣");
    tmpl.seed = get_date_seed();
    auto matrix = generate_random_matrix(tmpl.seed);
    if (verbose) {
        cout << "種子: " << tmpl.seed << endl;
        cout << "矩陣尺寸: 128x128" << endl;
        cout << "矩陣樣本 [0][0:5]: ";
        for (int i = 0; i < 5; i++) cout << fixed << setprecision(4) << matrix[0][i] << " ";
        cout << endl;
    }
    
    // Step 5: BioHash 投影
    if (verbose) print_line("Step 5: BioHash 投影");
    vector<float> biohash = biohash_projection(feature, matrix);
    if (verbose) {
        cout << "投影結果維度: " << biohash.size() << endl;
        cout << "投影結果: ";
        print_float_array(biohash);
    }
    
    // Step 6: 二值化
    if (verbose) print_line("Step 6: 二值化");
    float median;
    vector<uint8_t> all_bits = binarize(biohash, median);
    if (verbose) {
        cout << "中位數閾值: " << fixed << setprecision(4) << median << endl;
        cout << "二值化結果 (128 bits): ";
        print_bits(all_bits);
        int ones = count(all_bits.begin(), all_bits.end(), 1);
        cout << "1的數量: " << ones << ", 0的數量: " << (128 - ones) << endl;
    }
    
    // Step 7: Reliable Bit Selection
    if (verbose) print_line("Step 7: Reliable Bit Selection");
    vector<uint8_t> selected_bits;
    vector<float> selected_magnitudes;
    select_reliable_bits(biohash, all_bits, selected_bits, tmpl.indices, selected_magnitudes);
    if (verbose) {
        cout << "選擇的位元數: " << BIOHASH_K << endl;
        cout << "選擇的索引: [";
        for (int i = 0; i < BIOHASH_K; i++) {
            cout << tmpl.indices[i];
            if (i < BIOHASH_K - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        cout << "對應的投影值絕對值 (穩定度): [";
        for (int i = 0; i < BIOHASH_K; i++) {
            cout << fixed << setprecision(3) << selected_magnitudes[i];
            if (i < BIOHASH_K - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        cout << "選擇的位元: ";
        print_bits(selected_bits);
    }
    
    // Step 8: BCH 編碼
    if (verbose) print_line("Step 8: BCH 編碼");
    tmpl.codeword = bch_encode(selected_bits, tmpl.ecc_bytes_len);
    if (verbose) {
        cout << "BCH 參數: BCH(255, " << BIOHASH_K << ", t=" << BCH_T << ")" << endl;
        cout << "數據位元: " << BIOHASH_K << " bits (" << BIOHASH_K_BYTES << " bytes)" << endl;
        cout << "校驗位: " << (tmpl.ecc_bytes_len * 8) << " bits (" << tmpl.ecc_bytes_len << " bytes)" << endl;
        cout << "碼字總長: " << tmpl.codeword.size() << " bytes" << endl;
        cout << "碼字 (hex): " << bytes_to_hex(tmpl.codeword) << endl;
    }
    
    // Step 9: 生成最終模板
    if (verbose) print_line("Step 9: 最終模板");
    string template_hex = tmpl.to_hex();
    if (verbose) {
        cout << "模板組成:" << endl;
        cout << "  - 種子: 4 bytes" << endl;
        cout << "  - ECC長度: 1 byte" << endl;
        cout << "  - 索引: " << BIOHASH_K << " bytes" << endl;
        cout << "  - 碼字: " << tmpl.codeword.size() << " bytes" << endl;
        cout << "  - 總計: " << (4 + 1 + BIOHASH_K + tmpl.codeword.size()) << " bytes" << endl;
        cout << "\n模板 (hex):\n" << template_hex << endl;
    }
    
    return tmpl;
}

// ==================== 驗證函數 ====================
bool verify_image_with_template(const string& image_path, const BioHashTemplate& tmpl, bool verbose = true) {
    if (verbose) print_line("驗證開始");
    
    // 載入圖片
    Mat img = imread(image_path);
    if (img.empty()) {
        cerr << "Error: Cannot load image" << endl;
        return false;
    }
    if (verbose) cout << "載入圖片: " << image_path << endl;
    
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    
    // 人臉檢測
    MtcnnDetector detector("../models");
    vector<FaceInfo> results = detector.Detect(ncnn_img);
    if (results.empty()) {
        cerr << "Error: No face detected!" << endl;
        return false;
    }
    if (verbose) cout << "人臉檢測完成" << endl;
    
    // 人臉對齊
    ncnn::Mat aligned = preprocess(ncnn_img, results[0]);
    
    // 特徵提取
    Arcface arc("../models");
    vector<float> feature = arc.getFeature(aligned);
    if (verbose) cout << "特徵提取完成" << endl;
    
    // 使用模板中的種子生成矩陣
    if (verbose) cout << "使用種子: " << tmpl.seed << endl;
    auto matrix = generate_random_matrix(tmpl.seed);
    
    // BioHash 投影
    vector<float> biohash = biohash_projection(feature, matrix);
    if (verbose) cout << "BioHash 投影完成" << endl;
    
    // 二值化
    float median;
    vector<uint8_t> all_bits = binarize(biohash, median);
    if (verbose) cout << "二值化完成" << endl;
    
    // 使用模板中的索引提取位元
    vector<uint8_t> selected_bits(BIOHASH_K);
    for (int i = 0; i < BIOHASH_K; i++) {
        selected_bits[i] = all_bits[tmpl.indices[i]];
    }
    if (verbose) {
        cout << "使用模板索引提取位元" << endl;
        cout << "提取的位元: ";
        print_bits(selected_bits);
    }
    
    // BCH 解碼驗證
    if (verbose) print_line("BCH 解碼驗證");
    vector<uint8_t> corrected_bits;
    int num_errors;
    bool success = bch_decode_and_verify(selected_bits, tmpl.codeword, tmpl.ecc_bytes_len, 
                                          corrected_bits, num_errors);
    
    if (verbose) {
        if (success) {
            cout << "解碼成功！" << endl;
            cout << "錯誤數: " << num_errors << endl;
            cout << "糾正後位元: ";
            print_bits(corrected_bits);
        } else {
            cout << "解碼失敗！錯誤數超過 " << BCH_T << endl;
        }
    }
    
    return success;
}

// ==================== 主程序 ====================
void print_usage(const char* prog) {
    cout << "用法:" << endl;
    cout << "  " << prog << " <image>              - 處理單張圖片，顯示完整流程" << endl;
    cout << "  " << prog << " <image> <template>   - 驗證圖片與模板" << endl;
    cout << endl;
    cout << "範例:" << endl;
    cout << "  " << prog << " ../image/image.png" << endl;
    cout << "  " << prog << " ../image/image2.png 01359b1d17..." << endl;
}

int main(int argc, char* argv[]) {
    cout << "========================================" << endl;
    cout << "    BioHash-BCH 詳細測試工具" << endl;
    cout << "========================================" << endl;
    cout << "BCH 配置: BCH(255, " << BIOHASH_K << ", t=" << BCH_T << ")" << endl;
    cout << endl;
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    string image_path = argv[1];
    
    if (argc == 2) {
        // 單張圖片 - 顯示完整處理流程
        cout << "模式: 單張圖片處理（完整流程）" << endl;
        BioHashTemplate tmpl = process_single_image(image_path, true);
        
        if (!tmpl.codeword.empty()) {
            print_line("處理完成");
            cout << "\n可使用以下模板進行驗證:\n";
            cout << tmpl.to_hex() << endl;
        }
    } 
    else if (argc == 3) {
        // 圖片 + 模板驗證
        cout << "模式: 圖片與模板驗證" << endl;
        string template_hex = argv[2];
        
        cout << "模板長度: " << template_hex.length() << " 字元 (" << template_hex.length()/2 << " bytes)" << endl;
        
        BioHashTemplate tmpl = BioHashTemplate::from_hex(template_hex);
        cout << "解析模板:" << endl;
        cout << "  - 種子: " << tmpl.seed << endl;
        cout << "  - ECC長度: " << tmpl.ecc_bytes_len << " bytes" << endl;
        cout << "  - 索引數: " << tmpl.indices.size() << endl;
        cout << "  - 碼字長度: " << tmpl.codeword.size() << " bytes" << endl;
        
        bool result = verify_image_with_template(image_path, tmpl, true);
        
        print_line("最終結果");
        if (result) {
            cout << "✓ 驗證通過 - 是同一人" << endl;
        } else {
            cout << "✗ 驗證失敗 - 不是同一人" << endl;
        }
    }
    
    return 0;
}
