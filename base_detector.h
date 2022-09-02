/* 
 * @Author: zhanghao
 * @LastEditTime: 2022-09-02 17:13:00
 * @FilePath: /sgtls_s_725/base_detector.h
 * @LastEditors: zhanghao
 * @Description: 
 */
#pragma once

#include <memory>
#include <string>
#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "logging.h"
#include "types/camera_frame.h"


struct DetectorInitOptions
{
    std::string onnx_path;
    std::string engine_path;

    std::vector<std::string> engineInputTensorNames;
    std::vector<std::string> engineOutputTensorNames;

    int gpu_id;
    bool ues_fp16;
    int batch_size;
    
    float nms_threshold;
    float score_threshold;

    int input_width;
    int input_height;
    int num_classes;
};


/* 
 * @description: Base Tensorrt Detector.
 * @author: zhanghao
 */
class BaseDetector
{
    public:
        BaseDetector() = default;
        virtual ~BaseDetector() = default;
    

        /* 
         * @description: Init the detector by options.
         * @param {DetectorInitOptions} &options
         * @return {status}
         * @author: zhanghao
         */
        virtual bool Init(const DetectorInitOptions &options) = 0;


        /* 
         * @description: Do infrence.
         * @param {CameraFrame*} camera_frame
         * @return {status}
         * @author: zhanghao
         */
        virtual bool Detect(CameraFrame* camera_frame) = 0;
        

        /* 
         * @description: Name of this detector
         * @return {Detector name}
         * @author: zhanghao
         */
        virtual std::string Name() const = 0;


        BaseDetector(const BaseDetector &) = delete;
        BaseDetector &operator=(const BaseDetector &) = delete;


    protected:
        bool _ParseOnnxToEngine();
        bool _SerializeEngineToFile();
        bool _DeserializeEngineFromFile();


    protected:
        DetectorInitOptions m_options_;
        Logger m_logger_;

        nvinfer1::ICudaEngine* m_engine_;
        nvinfer1::IExecutionContext* m_context_;
};


