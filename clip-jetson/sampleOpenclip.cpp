
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "argsParser.h"
#include "common.h"
#include "logger.h"
#include "buffers.h"
#include "parserOnnxConfig.h"
#include "opencv2/opencv.hpp"
#include "cnpy.h"

using namespace nvinfer1;
const std::string gSampleName = "OPENCLIP.distill_model";
using samplesCommon::SampleUniquePtr;

class SampleCLIPDistilledModel {
    public :

        SampleCLIPDistilledModel(const samplesCommon::OnnxSampleParams& params):
            mParams(params),
            mEngine(nullptr)
        {
        }

        // 构建执行环境以及网络模型
        bool build();

        bool infer();

    private :
        samplesCommon::OnnxSampleParams mParams;    // 测试项目的参数

        nvinfer1::Dims mInputDims;
        nvinfer1::Dims mOutputDims;

        std::string text_embeddings_npy_path = "./data/text_embedings.npy";  // 文本向量存储路径

        std::shared_ptr<nvinfer1::ICudaEngine> mEngine; 

        // 解析onnx模型，创建TensorRT模型
        bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, 
            SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
            SampleUniquePtr<nvonnxparser::IParser>& parser);

        // 处理数据输入
        bool processInput(const samplesCommon::BufferManager& buffers);

        // 输出结果
        bool verifyOutput(const samplesCommon::BufferManager& buffers);

};


/*
    解析onnx模型，创建TensorRT模型

    builder :  负责创建和TensorRT网络的创建过程；
    network :  用于定义神经网络的结构与层次
    config  :  用于配置构建引擎时的各种参数和选项

    这里简单介绍下创建和使用TensorRT引擎的几个步骤：
        1.创建网络定义并构建引擎：
            -- 使用 IBuilder 和 INetworkDefinition 定义神经网络的结构；
            -- 使用 IBuilderConfig 配置引擎构建选项
            -- 调用 IBuilder 的 buildEngineWithConfig 或者 buildSerializationNetwork 方法生成引擎
                ---- buildSerializationNetwork 将网络定义和配置编译成序列化的引擎（IHostMemory）。它返回的是一个序列化的内存块，可以被保存到文件中，也可以用于后续的加载和执行。
                ---- buildEngineWithConfig 直接构建一个可用于推理的引擎（IEngine），通常会在内部管理序列化。它简化了构建过程，使得在使用配置的同时创建引擎。
        2.执行推理：
            -- 创建 IExecutionContext 执行引擎推理的上下文
            -- 将输入数据复制到设备内存中，执行推理操作，然后将输出数据复制回主机内存
*/
bool SampleCLIPDistilledModel::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, 
            SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
            SampleUniquePtr<nvonnxparser::IParser>& parser) {
    
    // 解析ONNX模型文件，注意parser是与network绑定的，解析好的网络会直接传递给network
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    if (mParams.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    if (mParams.int8) {
        config->setFlag(BuilderFlag::kINT8);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

bool SampleCLIPDistilledModel::verifyOutput(const samplesCommon::BufferManager& buffers) {
    float* image_embeddings = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    
    /*
        计算概率
    */
    std::vector<size_t> shape = {static_cast<size_t>(mOutputDims.d[0]), static_cast<size_t>(mOutputDims.d[1])};
    cnpy::npy_save("test.npy", image_embeddings, shape, "w");

    return true;
}

bool SampleCLIPDistilledModel::build() {
    

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = 
        SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    // 解析onnx模型，并根据参数决定是否启用fp16或者int8模式，以及是否开启DLA核心加速
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream(); 
    if (!profileStream) {
        return false;
    }

    // config->setProfileStream(profileStream.get());
    config->setProfileStream(*profileStream);

    // 构建序列化网络
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    // IRuntiume 主要负责加载序列化的引擎，处理序列化和反序列化过程, 创建 ICudaEngine
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }

    // 构建ICudaEngine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter()
    );
    if (!mEngine) {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    // 确定输入和输出的tensorname
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
        const char* bindingName = mEngine->getBindingName(i);
        
        if (mEngine->bindingIsInput(i)) {
            mParams.inputTensorNames.push_back(bindingName);
        }
        else {
            mParams.outputTensorNames.push_back(bindingName);
        }
    }

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}


bool SampleCLIPDistilledModel::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context) {
        return false;
    }

    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers)) {
        return false;
    }

    // 移动数据到device memory
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }

    buffers.copyOutputToHost();

    if (!verifyOutput(buffers)) {
        return false;
    }

    return true;
}

bool SampleCLIPDistilledModel::processInput(const samplesCommon::BufferManager& buffers) {

    const float IMAGE_MEAN[3] = {0.485, 0.456, 0.406};
    const float IMAGE_STD[3] = {0.229, 0.224, 0.225};

    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // Load Image and preprocess
    cv::Mat image = cv::imread("./data/test.png");
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

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    // convert to NCHW format
    int volch1 = inputH * inputW;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < inputH; ++h) {
            for (int w = 0; w < inputW; ++w) {
                hostDataBuffer[c * volch1 + h * inputW + w] = resizeImage.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return true;
}

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args) {
    samplesCommon::OnnxSampleParams params;

    if (args.dataDirs.empty())
    {
        params.dataDirs.push_back("./data/");
    }
    else 
    {
        params.dataDirs = args.dataDirs;
    }

    params.onnxFileName = "resnet18.onnx";
    // params.inputTensorName.push_back("");
    // params.outputTensorName.push_back("");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;

} 

void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/images/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


int main(int argc, char **argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK) {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }

    if (args.help) {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleCLIPDistilledModel sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and runing a GPU inference for CLIP Distilled model" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}