#ifndef CLIP_UTILS_H
#define CLIP_UTILS_H

#include <vector>
#include <string>
#include <cnpy.h>
#include <opencv2/opencv.hpp>

const float IMAGE_MEAN[3] = {0.485, 0.456, 0.406};
const float IMAGE_STD[3] = {0.229, 0.224, 0.225};

std::vector<std::vector<float>> embedding_to_probs(const std::vector<std::vector<float>>& embedding,
             const std::vector<std::vector<float>>& text_embedding, 
             float temp = 100.0f);

template <typename T>
std::vector<T> npyToVector(const cnpy::NpyArray& npyArray);

template <typename T>
std::vector<std::vector<T>> npyTo2DVector(const cnpy::NpyArray& npyArray);

template <typename T>
std::vector<std::vector<std::vector<T>>> npyTo3DVector(const cnpy::NpyArray& npyArray);

std::vector<std::string> readFileByLines(const std::string& filename);

void save2DNpy(const std::string& filename, const std::vector<std::vector<float>>& matrix);

void preprocess(const cv::Mat& image, float* pImage, int inputH, int inputW);

#endif
