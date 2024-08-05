#include "utils.h"
#include <opencv2/opencv.hpp>

// 图像预处理
void preprocess(const cv::Mat& image, float* pImage, int inputH, int inputW)
{
    cv::Mat resizeImage, floatImage;
    cv::resize(image, resizeImage, cv::Size(inputW, inputH));
    resizeImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    //Normalize
    std::vector<cv::Mat> channels(3);
    cv::split(resizeImage, channels);
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - IMAGE_MEAN[c]) / IMAGE_STD[c];
    }
    cv::merge(channels, resizeImage);

    // convert to NCHW format
    int volch1 = inputH * inputW;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < inputH; ++h) {
            for (int w = 0; w < inputW; ++w) {
                pImage[c * volch1 + h * inputW + w] = resizeImage.at<cv::Vec3f>(h, w)[c];
            }
        }
    }


}