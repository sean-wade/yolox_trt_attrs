/* 
 * @Author: zhanghao
 * @LastEditTime: 2022-09-02 17:20:12
 * @FilePath: /sgtls_s_725/detector_yolox.h
 * @LastEditors: zhanghao
 * @Description: 
 */
#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "logging.h"
#include "base_detector.h"
#include "detector_utils.h"
#include "types/camera_frame.h"
#include "types/object_detected.h"


struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};


class DetectorYoloX : public BaseDetector
{
    public:
        DetectorYoloX();
        ~DetectorYoloX();
    
        /* 
         * @description: 
         * @param {DetectorInitOptions} &options
         * @return {Init status}
         * @author: zhanghao
         */
        virtual bool Init(const DetectorInitOptions &options);

        /* 
         * @description: 
         * @param {CameraFrame*} camera_frame
         * @return {Process status}
         * @author: zhanghao
         */
        virtual bool Detect(CameraFrame* camera_frame);

        /* 
         * @description: 
         * @param {Mat*} image_ptr
         * @param {DetObjectList&} results
         * @return {*}
         * @author: zhanghao
         */
        virtual bool Detect(cv::Mat* image_ptr, DetObjectList& results);

        /* 
         * @description: Name of this detector
         * @return {Detector name}
         * @author: zhanghao
         */
        virtual std::string Name() const {
            return "sgtls_s_725";
        };


    private:
        bool _InitEngine();
        bool _DoInference();
        void _Preprocess(cv::Mat* img);
        
        void _GetOutputSize();
        void _GenerateGridsAndStride();
        void _DecodeOutputs(DetObjectList& result_objs);
        void _GenerateProposals(DetObjectList& prop_objs);
        void _GetAttrsFromFeaturemap(float* prob, DetObject& obj);
        
    private:
        int m_output_dim_size_;

        unsigned int m_img_channel_;
        
        float* m_input_blob_;
        float* m_output_prob_;

        unsigned int m_input_w_;
        unsigned int m_input_h_;

        int m_ori_img_w_;
        int m_ori_img_h_;
        float m_scale_;

        std::vector<int> m_strides_;
        std::vector<GridAndStride> m_grid_strides_;

        int m_num_classes_;
        int m_attr_channel_num_;
        std::vector<int> m_attr_num_list_;
};

