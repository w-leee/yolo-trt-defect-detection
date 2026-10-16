#include "trt_engine.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// 日志器
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

TRTEngine::TRTEngine(const std::string& engine_path) {
    // 读 engine 文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("无法打开 engine: " + engine_path);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    // 反序列化
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) {
        throw std::runtime_error("反序列化 engine 失败");
    }
    context_ = engine_->createExecutionContext();

    // 动态 shape 必须先设置输入 shape
    nvinfer1::Dims4 input_dims{1, 3, input_h_, input_w_};
    context_->setInputShape("images", input_dims);

    // 分配输入显存
    cudaMalloc(&gpu_input_, 1 * 3 * input_h_ * input_w_ * sizeof(float));

    // 从 context 拿输出 shape
    auto out_dims = context_->getTensorShape("output0");
    std::cout << "[DEBUG] 输出维度数: " << out_dims.nbDims << std::endl;
    output_size_ = 1;
    for (int i = 0; i < out_dims.nbDims; i++) {
        std::cout << "[DEBUG] 维度 " << i << ": " << out_dims.d[i] << std::endl;
        output_size_ *= out_dims.d[i];
    }

    // 兜底：如果还是负的，硬编码
    if (output_size_ <= 0) {
        output_size_ = 5 * num_anchors_;
        std::cout << "[WARN] 从 context 拿不到输出 shape，硬编码为 " << output_size_ << std::endl;
    }

    cudaMalloc(&gpu_output_, output_size_ * sizeof(float));

    // 创建 stream
    cudaStreamCreate(&stream_);

    std::cout << "[TRTEngine] 加载成功，输入 "
              << input_w_ << "x" << input_h_
              << "，输出元素数 " << output_size_ << std::endl;
}

TRTEngine::~TRTEngine() {
    if (stream_) cudaStreamDestroy(stream_);
    if (gpu_input_) cudaFree(gpu_input_);
    if (gpu_output_) cudaFree(gpu_output_);
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

void TRTEngine::preprocess(const cv::Mat& img, float* gpu_input) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w_, input_h_));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input_data(3 * input_h_ * input_w_);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_h_; h++) {
            for (int w = 0; w < input_w_; w++) {
                input_data[c * input_h_ * input_w_ + h * input_w_ + w] =
                    rgb.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }

    cudaMemcpyAsync(gpu_input, input_data.data(),
                    3 * input_h_ * input_w_ * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
}

float TRTEngine::iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<Detection> TRTEngine::postprocess(const float* output,
                                                int img_w, int img_h,
                                                float conf_thresh,
                                                float iou_thresh) {
    std::vector<Detection> detections;

    for (int i = 0; i < num_anchors_; i++) {
        float conf = output[4 * num_anchors_ + i];
        if (conf < conf_thresh) continue;

        float cx = output[0 * num_anchors_ + i];
        float cy = output[1 * num_anchors_ + i];
        float w  = output[2 * num_anchors_ + i];
        float h  = output[3 * num_anchors_ + i];

        Detection d;
        d.x1 = (cx - w / 2) * img_w / input_w_;
        d.y1 = (cy - h / 2) * img_h / input_h_;
        d.x2 = (cx + w / 2) * img_w / input_w_;
        d.y2 = (cy + h / 2) * img_h / input_h_;
        d.conf = conf;
        d.class_id = 0;
        detections.push_back(d);
    }

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (!suppressed[j] && iou(detections[i], detections[j]) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

std::vector<Detection> TRTEngine::infer(const cv::Mat& img,
                                          float conf_thresh,
                                          float iou_thresh) {
    // 预处理
    preprocess(img, static_cast<float*>(gpu_input_));

    // 确保输入 shape 已设置
    nvinfer1::Dims4 input_dims{1, 3, input_h_, input_w_};
    context_->setInputShape("images", input_dims);

    // 设置输入输出地址
    context_->setTensorAddress("images", gpu_input_);
    context_->setTensorAddress("output0", gpu_output_);

    // 推理
    context_->enqueueV3(stream_);

    // 拷回输出
    std::vector<float> output_data(output_size_);
    cudaMemcpyAsync(output_data.data(), gpu_output_,
                    output_size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return postprocess(output_data.data(), img.cols, img.rows,
                       conf_thresh, iou_thresh);
}