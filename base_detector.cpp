/* 
 * @Author: zhanghao
 * @LastEditTime: 2022-09-02 18:51:33
 * @FilePath: /sgtls_s_725/base_detector.cpp
 * @LastEditors: zhanghao
 * @Description: 
 */
#include "base_detector.h"


bool BaseDetector::_ParseOnnxToEngine()
{
    auto _builder = nvinfer1::createInferBuilder(m_logger_);
    if (!_builder)
    { 
        printf("Builder not created !");
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto _network = _builder->createNetworkV2(explicitBatch);
    if (!_network)
    {
        printf("Network not created ! ");
        return false;
    }

    auto _config = _builder->createBuilderConfig();
    if (!_config)
    {
        printf("Config not created ! ");
        return false;
    }

    auto _parser = nvonnxparser::createParser(*_network, m_logger_);
    if (!_parser)
    {
        printf("Parser not created ! ");
        return false;
    }
    printf("ConstructNetwork !\n");

    auto parsed = _parser->parseFromFile(m_options_.onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed)
    {
        printf("Onnx model cannot be parsed ! ");
        return false;
    }

    _builder->setMaxBatchSize(m_options_.batch_size);
    _config->setMaxWorkspaceSize(8 * (1 << 20));     // 8GB
    if (m_options_.ues_fp16)
    {
        _config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    printf("buildEngineWithConfig !\n");

    m_engine_ = _builder->buildEngineWithConfig(*_network, *_config);

    if (!m_engine_)
    {
        printf("Engine cannot be built ! ");
        return false;
    }
    
    printf("Create Engine  !\n");

    _parser->destroy();
    _network->destroy();
    _config->destroy();
    _builder->destroy();

    _SerializeEngineToFile();

    // reload model, there's a bug if not reload.
    m_engine_->destroy();
    if(!_DeserializeEngineFromFile())
    {
        printf("Engine rebuild failed!");
        return false;
    }

    return true;
}


bool BaseDetector::_SerializeEngineToFile()
{
    if(m_options_.engine_path == "") 
    {
        printf("Empty engine file name, skip save");
        return false;
    }
    
    if(m_engine_ != nullptr) 
    {
        printf("Saving engine to %s...", m_options_.engine_path.c_str());
        nvinfer1::IHostMemory* modelStream = m_engine_->serialize();
        std::ofstream file;
        file.open(m_options_.engine_path, std::ios::binary | std::ios::out);
        if(!file.is_open()) 
        {
            printf("Cannot write to engine file {}!");
            return false;
        }
        file.write((const char*)modelStream->data(), modelStream->size());
        file.close();
        modelStream->destroy();
    } 
    else 
    {
        printf("Engine is empty, save engine failed");
        return false;
    }

    printf("Saving engine succeed.");
    return true;
}


bool BaseDetector::_DeserializeEngineFromFile()
{
    cudaSetDevice(m_options_.gpu_id);
    char *_trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(m_options_.engine_path, std::ios::binary);
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        _trtModelStream = new char[size];
        assert(_trtModelStream);
        file.read(_trtModelStream, size);
        file.close();
    }
    else
    {
        printf("Cannot open engine path [%s]!\n", m_options_.engine_path.c_str());
        return false;
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(m_logger_);
    if(!runtime) 
    {
        return false;
    }

    m_engine_ = runtime->deserializeCudaEngine(_trtModelStream, size);
    if(!m_engine_) 
    {
        return false;
    }

    m_context_ = m_engine_->createExecutionContext();
    if(!m_context_) 
    {
        return false;
    }
    
    delete[] _trtModelStream;

    runtime->destroy();
    return true;
}