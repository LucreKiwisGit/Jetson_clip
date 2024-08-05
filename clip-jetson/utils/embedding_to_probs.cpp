#include "utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

std::vector<std::vector<float>> normalize(const std::vector<std::vector<float>>& matrix) {
    std::vector<std::vector<float>> normalized_matrix(matrix.size(), std::vector<float>(matrix[0].size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            norm += matrix[i][j] * matrix[i][j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            normalized_matrix[i][j] = matrix[i][j] / norm;
        }
    }
    return normalized_matrix;
}

std::vector<std::vector<float>> matrix_multiply(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t inner_dim = B.size();
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner_dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& logits, float temp) {
    std::vector<std::vector<float>> softmax_result(logits.size(), std::vector<float>(logits[0].size()));
    for (size_t i = 0; i < logits.size(); ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < logits[i].size(); ++j) {
            softmax_result[i][j] = std::exp(logits[i][j] * temp);
            sum += softmax_result[i][j];
        }
        for (size_t j = 0; j < logits[i].size(); ++j) {
            softmax_result[i][j] /= sum;
        }
    }
    return softmax_result;
}

std::vector<std::vector<float>> embedding_to_probs(const std::vector<std::vector<float>>& embedding, const std::vector<std::vector<float>>& text_embedding, float temp) {
    auto normalized_embedding = normalize(embedding);
    auto normalized_text_embedding = normalize(text_embedding);
    auto logits = matrix_multiply(normalized_embedding, normalized_text_embedding);
    return softmax(logits, temp);
}

