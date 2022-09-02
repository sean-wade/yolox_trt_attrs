/* 
 * @Author: zhanghao
 * @LastEditTime: 2022-09-02 18:34:51
 * @FilePath: /sgtls_s_725/main.cpp
 * @LastEditors: zhanghao
 * @Description: 
 */
#include "detector_yolox.h"

int main1()
{
    // FYI, diffrent opencv build options may cause diffrent speed.
    // i.e. on my pc(opencv3.2.0), resize function using 60+ms, yet on jarvis using 1ms
    // std::cout << cv::getBuildInformation() << std::endl;
    
    DetectorInitOptions options;
    options.input_width = 1024;
    options.input_height = 576;
    options.num_classes = 1;
    options.batch_size = 1;
    options.engineInputTensorNames = {"input_0"};
    options.engineOutputTensorNames = {"output_0"};
    options.engine_path = "../sgtls_s_725.engine";
    options.score_threshold = 0.5;
    options.nms_threshold = 0.25;
    options.ues_fp16 = false;

    DetectorYoloX yolox_det;
    std::cout << "Detecor name = " << yolox_det.Name() << std::endl;
    
    bool status = yolox_det.Init(options);
    std::cout << "Init status = " << status << std::endl;

    // cv::Mat image = cv::imread("/mnt/data/SGTrain/000_20220801/f30_undist/1655361292.500000000.jpg");
    // cv::Mat image = cv::imread("/home/zhanghao/code/GitLab/traffic_light_pipeline_py/data/test_seq/demo/gs1/1661418642.000000000.jpg");
    cv::Mat image = cv::imread("/home/zhanghao/code/GitLab/traffic_light_pipeline_py/data/test_seq/demo/gs1/1661418670.100000143.jpg");
    // CameraFrame camera_frame;
    // camera_frame.camera_image = std::shared_ptr<cv::Mat>(&image);

    // std::cout << "Start detect." << std::endl;

    // yolox_det.Detect(&camera_frame);

    auto start = std::chrono::system_clock::now();

    DetObjectList results;
    yolox_det.Detect(&image, results);

    std::cout << "Detect obj nums = " << results.size() << std::endl;

    auto end = std::chrono::system_clock::now();
    std::cout << "Total time using: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    draw_objects_save(image, results, "det_res_main.jpg");

    return 0;
}


int main()
{
    DetectorInitOptions options;
    options.input_width = 1024;
    options.input_height = 576;
    options.num_classes = 1;
    options.batch_size = 1;
    options.engineInputTensorNames = {"images"};
    options.engineOutputTensorNames = {"output"};
    options.onnx_path = "../models/sgtls_s_725.onnx";
    options.engine_path = "../models/sgtls_s_725.engine";
    options.score_threshold = 0.5;
    options.nms_threshold = 0.25;
    options.gpu_id = 0;
    options.ues_fp16 = false;

    DetectorYoloX yolox_det;
    std::cout << "Detecor name = " << yolox_det.Name() << std::endl;
    
    bool status = yolox_det.Init(options);
    std::cout << "Init status = " << status << std::endl;

    cv::Mat image = cv::imread("/home/zhanghao/code/GitLab/traffic_light_pipeline_py/data/test_seq/demo/gs1/1661418670.100000143.jpg");
    
    CameraFrame camera_frame;
    camera_frame.image_ptr = &image;

    std::cout << "Start detect." << std::endl;
    auto start = std::chrono::system_clock::now();

    for(int jj = 0; jj<100; jj++)
    {
        yolox_det.Detect(&camera_frame);
    }
    std::cout << "Detect obj nums = " << camera_frame.det_objects.size() << std::endl;

    auto end = std::chrono::system_clock::now();
    std::cout << "Avg time using: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/100 << "ms" << std::endl;

    draw_objects_save(image, camera_frame.det_objects, "det_res_main2.jpg");


    return 0;
}