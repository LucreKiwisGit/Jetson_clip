#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cnpy.h>
#include "utils.h"


// class Logger : public nvinfer1::ILogger
// {
//     void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
//     {
//         if (severity <= nvinfer1::ILogger::Severity::kWARNING)
//         {
//             std::cout << msg << std::endl;
//         }
//     }
// } gLogger;

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


int main()
{

    // 清理
    builder->destroy();

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
    size_t image_num = 1;
    inputDims.d[0] = image_num;
    outputDims.d[0] = image_num;
    for (int i = 0; i < inputDims.nbDims; ++i)
    {
        inputSize *= inputDims.d[i];
    }
    size_t outputSize = 1;
    for (int i = 1; i < outputDims.nbDims; ++i)
    {
        outputSize *= outputDims.d[i];
    }

    void* buffer[2];
    cudaMalloc(&buffer[0], inputSize * sizeof(float));
    cudaMalloc(&buffer[1], outputSize * sizeof(float));

    // Load Image and preprocess
    cv::Mat image = cv::imread("test.png");
    float* pImage = new float[inputSize];
    preprocess(image, pImage, inputDims.d[2], inputDims.d[3]);
    cudaMemcpy(buffer[0], pImage, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // 开始推理
    bool status = context->executeV2(buffer);
    if (!status) {
        std::cerr << "Inference execution failed." << std::endl;
        return 0;
    }


    // 把输出结果复制到CPU内存
    std::vector<std::vector<float>> embedding(outputDims.d[0], std::vector<float>(outputSize));
    cudaMemcpy(embedding.data(), buffer[1], outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 结果后处理
    std::vector<std::vector<float>> probs;
    std::vector<std::vector<float>> text_embedding;
    cnpy::NpyArray text_embedding_npy = cnpy::npy_load("./data/text_embeddings.npy");
    text_embedding = npyTo2DVector<float>(text_embedding_npy);
    probs = embedding_to_probs(embedding, text_embedding);
    
    // 输出结果
    save2DNpy("probs.npy", probs);
    std::vector<std::string> text_prompts = readFileByLines("./data/text_prompts.txt");
    for (int i = 0; i < probs.size();i++)
    {
        printf("The %d 'th result:\n", i);
        for (int j = 0; j < text_prompts.size();j++)
        {
            printf("%s : %f", text_prompts[j], probs[i][j]);
        }
    }

    // 释放内存
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}