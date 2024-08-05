#include "utils.h"
#include "cnpy.h"
#include <string>
#include <cstring>
#include <iostream>

template <typename T>
std::vector<T> npyToVector(const cnpy::NpyArray& npyArray) {
    if (npyArray.word_size != sizeof(T)) {
        throw std::runtime_error("Datatype mismatch!");
    }

    size_t num_elements = npyArray.num_vals;
    std::vector<T> vec(num_elements);
    std::memcpy(vec.data(), npyArray.data<T>(), num_elements * sizeof(T));

    return vec;
}

template <typename T>
std::vector<std::vector<T>> npyTo2DVector(const cnpy::NpyArray& npyArray) {
    if (npyArray.word_size != sizeof(T)) {
        throw std::runtime_error("Datatype mismatch!");
    }

    size_t rows = npyArray.shape[0];
    size_t cols = npyArray.shape[1];

    std::vector<std::vector<T>> vec(rows, std::vector<T>(cols));
    std::memcpy(vec[0].data(), npyArray.data<T>(), rows * cols * sizeof(T));

    return vec;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> npyTo3DVector(const cnpy::NpyArray& npyArray) {
    if (npyArray.word_size != sizeof(T)) {
        throw std::runtime_error("Datatype mismatch!");
    }

    size_t dim1 = npyArray.shape[0];
    size_t dim2 = npyArray.shape[1];
    size_t dim3 = npyArray.shape[2];

    std::vector<std::vector<std::vector<T>>> vec(dim1, std::vector<std::vector<T>>(dim2, std::vector<T>(dim3)));
    std::memcpy(vec[0].data(), npyArray.data<T>(), dim1 * dim2 *dim3 * sizeof(T));

    return vec;
}

// 将二维 std::vector 保存为 .npy 文件
void save2DNpy(const std::string& filename, const std::vector<std::vector<float>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    // 创建一个一维数组来存储所有数据
    std::vector<float> data;
    data.reserve(rows * cols);

    // 将二维 std::vector 的数据展平到一维数组
    for (const auto& row : matrix) {
        data.insert(data.end(), row.begin(), row.end());
    }

    // 保存为 .npy 文件
    cnpy::npy_save(filename, data.data(), {rows, cols}, "w");
}

// 显式实例化模板函数
template std::vector<float> npyToVector<float>(const cnpy::NpyArray&);
template std::vector<std::vector<float>> npyTo2DVector<float>(const cnpy::NpyArray&);
template std::vector<std::vector<std::vector<float>>> npyTo3DVector<float>(const cnpy::NpyArray&);