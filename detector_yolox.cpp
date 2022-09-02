/* 
 * @Author: zhanghao
 * @LastEditTime: 2022-09-02 18:51:16
 * @FilePath: /sgtls_s_725/detector_yolox.cpp
 * @LastEditors: zhanghao
 * @Description: 
 */
#include <unistd.h>
#include "detector_yolox.h"


DetectorYoloX::DetectorYoloX()
{
    m_engine_ = nullptr;
    m_context_ = nullptr;

    m_img_channel_ = 3;
    m_strides_ = {8, 16, 32};
    m_num_classes_ = 1;
    m_attr_num_list_ = {
        4, 3, 3, 6, 4, 2, 2, 5, 2, 5, 14
    };

    m_attr_channel_num_ = 0;
    for(auto n : m_attr_num_list_){
        m_attr_channel_num_ += n;
    }
};


DetectorYoloX::~DetectorYoloX()
{
    if(m_context_)
        m_context_->destroy();
    
    if(m_engine_)
        m_engine_->destroy();

    if(m_input_blob_)
        delete m_input_blob_;

    if(m_output_prob_)
        delete m_output_prob_;

    // GPU_CHECK(cudaFreeHost(m_input_blob_));
    // GPU_CHECK(cudaFreeHost(m_output_prob_));
};


bool DetectorYoloX::Init(const DetectorInitOptions &options)
{
    m_options_ = options;

    m_input_w_ = m_options_.input_width;
    m_input_h_ = m_options_.input_height;

    if(!_InitEngine())
    {
        return false;
    }
    _GetOutputSize();   // get m_output_dim_size_

    m_input_blob_ = new float[3 * m_input_w_ * m_input_h_];
    m_output_prob_ = new float[m_output_dim_size_];
    
    // cudaMallocHost((void**)&m_input_blob_, 3 * m_input_w_ * m_input_h_ * sizeof(float));
    // cudaMallocHost((void**)&m_output_prob_, m_output_dim_size_ * sizeof(float));
    
    _GenerateGridsAndStride();

    return true;
}


bool DetectorYoloX::_InitEngine()
{
    bool engine_exists = access(m_options_.engine_path.c_str(), F_OK) == 0;
    if(engine_exists)
    {
        printf("Engine file exist. Build from serialized engine : [%s]\n", m_options_.engine_path.c_str());

        bool desrial_status = _DeserializeEngineFromFile();
        if(!desrial_status)
        {
            printf("Engine build failed!");
            return false;
        }

        printf("Engine build success!\n");
    }                       
    else
    {
        printf("Engine file dose not exist! Build from onnx! This will take a while...\n");
        printf("Onnx path: %s\n", m_options_.onnx_path.c_str());
        
        if(!_ParseOnnxToEngine())
        {
            printf("Onnx parse failed!\n");
            return false;
        }
        printf("Engine has been created ! \n");
    }
    
    return true;
}


bool DetectorYoloX::Detect(cv::Mat* image_ptr, DetObjectList& results)
{
    // auto t1 = std::chrono::system_clock::now();
    
    _Preprocess(image_ptr);

    // auto t2 = std::chrono::system_clock::now();
    // std::cout << "_Preprocess time using: "  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    if(!_DoInference())
    {
        return false;
    }

    // auto t3 = std::chrono::system_clock::now();
    // std::cout << "_DoInference time using: "   << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "ms" << std::endl;
    
    // save_prob_txt(m_output_prob_, "prob_main.txt");


    _DecodeOutputs(results);
    // auto t4 = std::chrono::system_clock::now();
    // std::cout << "_DecodeOutputs time using: "    << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms" << std::endl;

    return true;
}


bool DetectorYoloX::Detect(CameraFrame* camera_frame)
{
    _Preprocess(camera_frame->image_ptr);

    if(!_DoInference())
    {
        return false;
    }
    
    _DecodeOutputs(camera_frame->det_objects);
    return true;
}


/* 
 * @description: from cv::Mat to m_input_blob_
 * @param {cv::Mat*} img_ptr
 * @return {status}
 * @author: zhanghao
 */
void DetectorYoloX::_Preprocess(cv::Mat* img_ptr) 
{
    m_ori_img_w_ = img_ptr->cols;
    m_ori_img_h_ = img_ptr->rows;
    m_scale_ = std::min(m_input_w_ / (img_ptr->cols*1.0), m_input_h_ / (img_ptr->rows*1.0));
    
    // 1. resize & padding
    int unpad_w = m_scale_ * img_ptr->cols;
    int unpad_h = m_scale_ * img_ptr->rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);

    auto t1 = std::chrono::system_clock::now();
    cv::resize(*img_ptr, re, re.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat out(m_input_h_, m_input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    // 2. copy data to blob
    for (size_t c = 0; c < m_img_channel_; c++) 
    {
        for (size_t  h = 0; h < m_input_h_; h++) 
        {
            for (size_t w = 0; w < m_input_w_; w++) 
            {
                m_input_blob_[c * m_input_w_ * m_input_h_ + h * m_input_w_ + w] =
                    (float)out.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}


/* 
 * @description: Tensorrt Excution.
 * @return {status}
 * @author: zhanghao
 */
bool DetectorYoloX::_DoInference()
{
    if(m_engine_->getNbBindings() != 2)
    {
        return false;
    }

    void* buffers[2];
    const int inputIndex = m_engine_->getBindingIndex(m_options_.engineInputTensorNames[0].c_str());
    
    if(m_engine_->getBindingDataType(inputIndex) != nvinfer1::DataType::kFLOAT)
    {
        return false;
    }   

    const int outputIndex = m_engine_->getBindingIndex(m_options_.engineOutputTensorNames[0].c_str());
    if(m_engine_->getBindingDataType(outputIndex) != nvinfer1::DataType::kFLOAT)
    {
        return false;
    }  
    // int mBatchSize = m_engine_->getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * m_input_h_ * m_input_w_ * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], m_output_dim_size_ * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], m_input_blob_, 3 * m_input_h_ * m_input_w_ * sizeof(float), cudaMemcpyHostToDevice, stream));
    m_context_->enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(m_output_prob_, buffers[outputIndex], m_output_dim_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    return true;
}


void DetectorYoloX::_GetOutputSize()
{
    m_output_dim_size_ = 1;
    auto out_dims = m_engine_->getBindingDimensions(1);
    for(int j=0; j<out_dims.nbDims; j++) 
    {
        m_output_dim_size_ *= out_dims.d[j];
    }
}


void DetectorYoloX::_GenerateGridsAndStride()
{
    for (auto stride : m_strides_)
    {
        int num_grid_y = m_input_h_ / stride;
        int num_grid_x = m_input_w_ / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                m_grid_strides_.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}


void DetectorYoloX::_GetAttrsFromFeaturemap(float* prob, DetObject& obj)
{
    // direction orientation state indication occlusion truncation blur child_num relevance color pict
    // 4, 3, 3, 6, 4, 2, 2, 5, 2, 5, 14

    obj.attributes.direction    =   TL_DIRECTION(   std::max_element(prob +  0, prob +  4) - prob -  0);
    obj.attributes.orientation  =   TL_ORIENTATION( std::max_element(prob +  4, prob +  7) - prob -  4);
    obj.attributes.state        =   TL_STATE(       std::max_element(prob +  7, prob + 10) - prob -  7);
    obj.attributes.indication   =   TL_INDICATION(  std::max_element(prob + 10, prob + 16) - prob - 10);
    obj.attributes.truncation   =   TL_TRUNCATION(  std::max_element(prob + 16, prob + 20) - prob - 16);
    obj.attributes.occlusion    =   TL_OCCLUSION(   std::max_element(prob + 20, prob + 22) - prob - 20);
    obj.attributes.blur         =   TL_BLUR(        std::max_element(prob + 22, prob + 24) - prob - 22);
    obj.attributes.child_num    =   TL_CHILD_NUM(   std::max_element(prob + 24, prob + 29) - prob - 24);
    obj.attributes.relevance    =   TL_RELEVANCE(   std::max_element(prob + 29, prob + 31) - prob - 29);
    obj.attributes.color        =   TL_COLOR(       std::max_element(prob + 31, prob + 36) - prob - 31);
    obj.attributes.pict         =   TL_PICT(        std::max_element(prob + 36, prob + 50) - prob - 36);

}


void DetectorYoloX::_GenerateProposals(DetObjectList& prop_objs)
{
    const int num_anchors = m_grid_strides_.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = m_grid_strides_[anchor_idx].grid0;
        const int grid1 = m_grid_strides_[anchor_idx].grid1;
        const int stride = m_grid_strides_[anchor_idx].stride;

        const int basic_pos = anchor_idx * (m_num_classes_ + 5 + m_attr_channel_num_);  // 1+5+50

        // std::cout << "basic_pos = " << basic_pos << "\n";

        // yolox/models/yolo_head.py decode logic
        float x_center = (m_output_prob_[basic_pos+0] + grid0) * stride;
        float y_center = (m_output_prob_[basic_pos+1] + grid1) * stride;
        float w = exp(m_output_prob_[basic_pos+2]) * stride;
        float h = exp(m_output_prob_[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = m_output_prob_[basic_pos+4];

        for (int class_idx = 0; class_idx < m_num_classes_; class_idx++)
        {
            float box_cls_score = m_output_prob_[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            
            if (box_prob > m_options_.score_threshold)
            {
                
                DetectObject obj;

                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.det_score = box_prob;

                _GetAttrsFromFeaturemap(m_output_prob_ + basic_pos + 6 + class_idx, obj);

                prop_objs.push_back(obj);
            }
        } // class loop
    } // point anchor loop
}


/* 
 * @description: Decode feature map to DetObjectList.
 * @param {DetObjectList&} result_objs
 * @return {*}
 * @author: zhanghao
 */
void DetectorYoloX::_DecodeOutputs(DetObjectList& result_objs)
{
    
    DetObjectList proposal_objects;
    _GenerateProposals(proposal_objects);

    // std::cout << "Num of boxes before nms: " << proposal_objects.size() << std::endl;

    qsort_descent_inplace(proposal_objects);
    std::vector<int> picked;
    nms_sorted_bboxes(proposal_objects, picked, m_options_.nms_threshold);

    int count = picked.size();
    // std::cout << "Num of boxes after nms: " << count << std::endl;

    result_objs.resize(count);
    for (int i = 0; i < count; i++)
    {
        result_objs[i] = proposal_objects[picked[i]];

        // adjust offset to original unpadded
        float x0 = (result_objs[i].rect.x) / m_scale_;
        float y0 = (result_objs[i].rect.y) / m_scale_;
        float x1 = (result_objs[i].rect.x + result_objs[i].rect.width) / m_scale_;
        float y1 = (result_objs[i].rect.y + result_objs[i].rect.height) / m_scale_;

        // clip
        x0 = std::max(std::min(x0, (float)(m_ori_img_w_ - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(m_ori_img_h_ - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(m_ori_img_w_ - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(m_ori_img_h_ - 1)), 0.f);

        result_objs[i].rect.x = x0;
        result_objs[i].rect.y = y0;
        result_objs[i].rect.width = x1 - x0;
        result_objs[i].rect.height = y1 - y0;
    }
}

