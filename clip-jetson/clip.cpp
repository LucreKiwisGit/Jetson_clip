#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <map>
#include <chrono>

class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
        {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// 载入模型
nvinfer1::ICudaEngine* loadModel(const std::string& engineFile, nvinfer1::IRuntime* runtime){
    std::ifstream file(engineFile, std::ios::binary);
    if (!file)
    {
        std::cout << "Could not find engine file." << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t length = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(length);
    file.read(engineData.data(), length);

    return runtime->deserializeCudaEngine(engineData.data(), length, nullptr);
}

// 图像预处理
cv::Mat preprocess(const cv::Mat& image, int inputH, int inputW)
{
    cv::Mat resizeImage, floatImage;
    cv::resize(image, resizeImage, cv::Size(inputW, inputH));
    resizeImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    return floatImage;
}

int main()
{
    // Load TensorRT runtime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
    {
        std::cout << "Failed to create TensorRT runtime." << std::endl;
        return -1;
    }

    // Load the TensorRT engine
    nvinfer1::ICudaEngine* engine = loadModel("resnet18.trt", runtime);
    if (!engine)
    {
        std::cout << "Failed to load TensorRT engine." << std::endl;
        return -1;
    }
    
    // Create execution context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cout << "Failed to create execution context." << std::endl;
        return -1;
    }

    // Load image
    int inputIndex = engine->getBindingIndex("input");
    int outputIndex = engine->getBindingIndex("output");
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

    // Allocate memory
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i)
    {
        inputSize *= inputDims.d[i];
    }
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i)
    {
        outputSize *= outputDims.d[i];
    }

    void* buffer[2];
    cudaMalloc(&buffer[0], inputSize * sizeof(float));
    cudaMalloc(&buffer[1], outputSize * sizeof(float));

    // Load Image and preprocess
    cv::Mat image = cv::imread("dog.jpg");
    cv::Mat preprocessedImage = preprocess(image, inputDims.d[1], inputDims.d[2]);

    cudaMemcpy(buffer[0], preprocessedImage.data, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    context->executeV2(buffer);

    // 把输出结果复制到CPU内存
    std::vector<float> embeding(outputSize);
    cudaMemcpy(output.data(), buffer[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 结果后处理
    

    // 释放内存
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}