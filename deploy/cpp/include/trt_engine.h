#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};

class TRTEngine {
public:
    TRTEngine(const std::string& engine_path);
    ~TRTEngine();

    std::vector<Detection> infer(const cv::Mat& img,
                                  float conf_thresh = 0.25f,
                                  float iou_thresh  = 0.45f);

private:
    void preprocess(const cv::Mat& img, float* gpu_input);
    std::vector<Detection> postprocess(const float* output, int img_w, int img_h,
                                        float conf_thresh, float iou_thresh);
    static float iou(const Detection& a, const Detection& b);

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    void* gpu_input_ = nullptr;
    void* gpu_output_ = nullptr;
    cudaStream_t stream_ = nullptr;

    int input_w_ = 416;
    int input_h_ = 416;
    int num_anchors_ = 3549;
    int num_classes_ = 1;
    int output_size_ = 0;
};