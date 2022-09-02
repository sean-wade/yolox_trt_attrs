/* 
 * @Author: zhanghao
 * @LastEditTime: 2022-09-02 15:49:33
 * @FilePath: /sgtls_s_725/detector_utils.h
 * @LastEditors: zhanghao
 * @Description: 
 */
#pragma once
#include <NvInfer.h>
#include "types/object_detected.h"


#define GPU_CHECK(ans)                                                                                                                               \
  {                                                                                                                                                                                 \
    GPUAssert((ans), __FILE__, __LINE__);                                                                                                 \
  }

inline void GPUAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


/////////////////////////////////////////////////////////////////////////////////////////////////////
//// sort & nms
/////////////////////////////////////////////////////////////////////////////////////////////////////

static void qsort_descent_inplace(DetObjectList& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].det_score;

    while (i <= j)
    {
        while (faceobjects[i].det_score > p)
            i++;

        while (faceobjects[j].det_score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(DetObjectList& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const DetObject& a, const DetObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const DetObjectList& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const DetObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const DetObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
//// txt save
/////////////////////////////////////////////////////////////////////////////////////////////////////

static void save_prob_txt(float* prob, std::string txt_path)
{
    std::fstream txt_file(txt_path, std::ios::out);

    for(int i=0; i<12096; i++)
    {
        for(int j=0; j<56; j++)
        {
            txt_file << prob[56*i+j] << " ";
        }
        txt_file << "\n";
    }
    txt_file.close();
}

static void save_input_txt(float* input, std::string txt_path)
{
    std::fstream txt_file(txt_path, std::ios::out);

    for(int i=0; i<3*1024*576; i++)
    {
        txt_file << input[i] << " ";
        txt_file << "\n";
    }
    txt_file.close();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
//// plot image
/////////////////////////////////////////////////////////////////////////////////////////////////////

static void draw_objects_save(const cv::Mat& bgr, const DetObjectList& objects, std::string save_path)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const DetObject& obj = objects[i];

        // fprintf(stderr, "obj : %.3f at %.1f %.1f %.1f x %.1f\n", obj.det_score,
        //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(0,255,255);
        cv::Scalar txt_color = cv::Scalar(0, 0, 0);

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%.1f", obj.det_score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y-label_size.height), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);


        char attr[256];
        sprintf(attr, "%d%d%d%d%d%d%d%d%d%d%d", 
                obj.attributes.direction,
                obj.attributes.orientation,
                obj.attributes.state,
                obj.attributes.indication,
                obj.attributes.truncation,
                obj.attributes.occlusion,
                obj.attributes.blur,
                obj.attributes.child_num,
                obj.attributes.relevance,
                obj.attributes.color,
                obj.attributes.pict
                );

        cv::putText(image, attr, cv::Point(x, y+obj.rect.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
    }

    cv::imwrite(save_path, image);
}
