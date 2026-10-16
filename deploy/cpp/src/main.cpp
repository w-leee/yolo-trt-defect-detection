#include "trt_engine.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0]
                  << " <engine> <image> [output]\n";
        return -1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "result.jpg";

    // 加载 engine
    TRTEngine engine(engine_path);

    // 读图
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "读不到图片: " << image_path << std::endl;
        return -1;
    }

    // ★ Warmup 10 次
    std::cout << "[Warmup] 10 次..." << std::endl;
    for (int i = 0; i < 10; i++) {
        engine.infer(img, 0.25f, 0.45f);
    }

    // ★ 测速 100 次
    std::cout << "[Benchmark] 100 次..." << std::endl;
    int runs = 100;
    std::vector<double> times;
    std::vector<Detection> detections;
    for (int i = 0; i < runs; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        detections = engine.infer(img, 0.25f, 0.45f);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // 统计
    double sum = 0, tmin = 1e9, tmax = 0;
    for (double t : times) {
        sum += t;
        if (t < tmin) tmin = t;
        if (t > tmax) tmax = t;
    }
    double avg = sum / runs;

    std::cout << "\n========== 结果 ==========\n";
    std::cout << "平均: " << avg << " ms\n";
    std::cout << "最小: " << tmin << " ms\n";
    std::cout << "最大: " << tmax << " ms\n";
    std::cout << "FPS:  " << 1000.0 / avg << "\n";
    std::cout << "检测到 " << detections.size() << " 个目标\n";

    // 画框
    for (const auto& d : detections) {
        cv::rectangle(img, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2),
                      cv::Scalar(0, 255, 0), 2);
        std::string label = "positive " + std::to_string(d.conf).substr(0, 4);
        cv::putText(img, label, cv::Point(d.x1, d.y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite(output_path, img);
    std::cout << "结果保存到: " << output_path << std::endl;

    return 0;
}