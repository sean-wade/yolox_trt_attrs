/* 
 * @Author: zhanghao
 * @LastEditTime: 2022-09-01 17:39:23
 * @FilePath: /sgtls_s_725/types/object_semantic.h
 * @LastEditors: zhanghao
 * @Description: 
 */
// /* 
//  * @Author: zhanghao
//  * @LastEditTime: 2022-08-31 10:45:58
//  * @FilePath: /camera_traffic_light_pipeline/src/include/types/object_semantic.h
//  * @LastEditors: zhanghao
//  * @Description: 
//  */

// #pragma once

// #include "box.h"
// #include "attributes.h"
// #include "semantic_type.h"
// #include "history_status.h"


// class SemanticLightObject
// {
//     public:
//         SemanticLightObject(double timestamp){
//             m_time_ = timestamp;
//             m_track_id_ = 0;
//             m_is_digit_ = false;
//             m_is_blink_ = false;
//             m_is_valid_ = false;
//             m_is_relevant_ = false;

//             m_det_color_ = TL_COLOR_UNKNOWN;
//             m_out_color_ = TL_COLOR_UNKNOWN;
//             m_semantic_ = SEMANTIC_TYPE_UNKNOWN;
            
//             m_digit_num_ = "";
//         };
//         ~SemanticLightObject();


//     public:
//         void CheckValid();
//         void UpdateFromTrackedObj();
//         void ReviseColorByHistory();


//     private:
//         void _Update_semantic();
//         void _AccColorChangeStreak();
//         void _AccColorChangeStreak();

//         void _JudgeBlink();
//         void _CommonRevise();
//         void _ReviseYellow();
//         void _ReviseUnknown();
//         void _ReviseDark();
//         void _ReviseRedGreen();
//         void _ReviseByHistoryVote();


//     private:
//         double m_time_;
//         unsigned int m_track_id_;
        
//         bool m_is_digit_;
//         bool m_is_blink_;
//         bool m_is_valid_;
//         bool m_is_relevant_;
//         float m_distance_;
//         TL_COLOR m_det_color_;
//         TL_COLOR m_out_color_;
//         SEMANTIC_TYPE m_semantic_;
//         BBox m_bbox_;

//         std::string m_digit_num_;

//         SemanticHistoryStatus m_history_;
// };
