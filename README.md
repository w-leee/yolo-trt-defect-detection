# 工业缺陷检测的 TensorRT 边缘推理加速

基于 **YOLOv8 + ONNX + TensorRT** 的工业缺陷实时检测与边缘推理优化。以**混凝土裂缝（crack）**为典型缺陷样本，完整打通从模型训练、ONNX 导出、TensorRT FP16/INT8 量化，到 C++ 高性能推理的全链路部署。

在 RTX 3050 Laptop（4GB 显存）这类典型边缘算力上，纯推理延迟从 PyTorch 的 **14.6 ms** 压缩到 TensorRT FP16 的 **3.72 ms**，C++ 端到端（预处理 + 推理 + 后处理）**4.84 ms / 207 FPS**，达到工业在线检测的实时性要求。

---

## 目录

- [项目亮点](#项目亮点)
- [性能基准](#性能基准)
- [模型精度](#模型精度)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用流程详解](#使用流程详解)
- [关键技术点](#关键技术点)
- [踩坑记录](#踩坑记录)
- [未来工作](#未来工作)
- [环境详情](#环境详情)
- [参考](#参考)

---

## 项目亮点

- **完整部署链路**：PyTorch → ONNX → TensorRT FP16/INT8 → C++ 推理，每一环都有可复现脚本与实测数据
- **推理加速约 3 倍**：PyTorch 14.6 ms → TensorRT FP16 3.72 ms（P50）
- **C++ 端到端 4.84 ms，FPS 207**：含预处理与后处理，非纯推理数字
- **INT8 量化**：模型体积减少 20%，权重内存减少 45%，并在量化受阻时给出明确归因
- **手写 NMS + DFL 解码**：不依赖 Ultralytics 后处理，C++ 侧零 Python 依赖
- **动态 shape 支持**：320×320 ~ 640×640 单引擎覆盖多分辨率

---

## 性能基准

| 阶段 | 延迟 | FPS | 模型大小 | 说明 |
|---|---|---|---|---|
| PyTorch (Python) | 14.6 ms | 68 | 6.2 MB | 纯推理 |
| ONNX Runtime (Python) | ~10 ms | ~100 | 12.5 MB | 未单独精细测量 |
| TensorRT FP16 (Python) | **3.72 ms** (P50) | 269 | 9.7 MB | 纯推理 |
| TensorRT INT8 (Python) | **3.53 ms** (P50) | 221 | 7.8 MB | 纯推理 |
| **TensorRT FP16 (C++ 端到端)** | **4.84 ms** | **207** | — | 预处理 + 推理 + 后处理 |

> **测试环境**：WSL2 Ubuntu + CUDA 12.6 + TensorRT 10.7 + RTX 3050 Laptop (4GB)
> 分辨率 416×416，warmup 10 次 + 100 次循环取 P50。

**结论**：相对 PyTorch 基线，FP16 引擎纯推理提速约 **3.9×**；C++ 端到端仍保持在 5 ms 以内，说明预处理与手写后处理不再是瓶颈，管线整体处于计算受限的最优区间。

---

## 模型精度

| 指标 | 值 |
|---|---|
| mAP50 | 0.847 |
| mAP50-95 | 0.825 |
| Precision | 0.661 |
| Recall | 0.942 |

模型为 **YOLOv8n**（nano），单类别缺陷检测。Recall 高达 0.942 是刻意取舍的结果——工业缺陷检测场景中漏检代价远高于误检，因此容忍一定误报换取高召回。

---

## 技术架构

```
┌─────────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────────┐
│  PyTorch    │───▶│  ONNX   │───▶│  TensorRT    │───▶│  C++ +       │
│  YOLOv8n    │    │ 12.5 MB │    │  FP16 / INT8 │    │  OpenCV      │
└─────────────┘    └─────────┘    └──────────────┘    └──────────────┘
    14.6 ms          ~10 ms           3.72 ms           4.84 ms
                                                        207 FPS
```

**数据流**：图像 → `letterbox` 缩放到 416×416 → 归一化 / HWC→CHW → GPU 拷贝 → TensorRT `enqueueV3` → 输出 `[1, 5, 3549]` → DFL 解码 + 框还原 → NMS → 绘制结果。

---

## 项目结构

```
yolo-demo/
├── crack_dataset/                 # 数据集（train 826 / valid 100 / test 50）
├── calib_images/                  # INT8 校准集（200 张）
├── scripts/                       # Python 工具脚本
│   ├── check_dataset.py           # 数据集校验
│   └── train.py                   # YOLOv8n 训练
├── weights/
│   ├── pretrained/                # 官方预训练权重
│   ├── trained/                   # 自己训练的模型（best.pt）
│   └── engines/                   # ONNX + FP16 + INT8 engine
├── deploy/
│   ├── trt_python/                # TensorRT Python 脚本
│   │   ├── onnx_to_engine.py      # FP16 engine 构建
│   │   ├── onnx_to_int8_engine.py # INT8 engine 构建
│   │   ├── int8_calibrator.py     # INT8 校准器
│   │   ├── trt_infer.py           # 推理性能测试
│   │   └── compare_fp16_int8.py   # FP16 vs INT8 对比
│   ├── cpp/                       # C++ TensorRT 推理
│   │   ├── CMakeLists.txt
│   │   ├── include/trt_engine.h
│   │   └── src/
│   │       ├── trt_engine.cpp     # 引擎封装：预处理 + 推理 + NMS
│   │       └── main.cpp           # 主程序
│   ├── engine/                    # 部署用 engine 副本
│   └── docs/                      # 性能报告 + 结果图
│       ├── benchmark_cpp.md
│       └── cpp_result.jpg
├── runs/detect/train/             # 训练曲线、混淆矩阵
├── third_party/                   # TensorRT C++ SDK
├── ENVIRONMENT.md                 # 环境配置文档
└── README.md
```

---

## 快速开始

### 环境要求

| 组件 | 版本 |
|---|---|
| 系统 | WSL2 Ubuntu 22.04 或原生 Ubuntu |
| CUDA | 12.6 |
| TensorRT | 10.7 |
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| OpenCV | 4.10 |

> Python 版本请严格使用 3.10。3.14 过新，PyTorch 生态尚无对应 wheel，会直接导致安装失败（详见[踩坑记录](#踩坑记录)）。

### 创建虚拟环境

```bash
cd ~/workspace/yolo_trt/yolo-demo
python3.10 -m venv venv
source venv/bin/activate
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics onnx onnxsim onnxruntime-gpu opencv-python
pip install tensorrt==10.7 --no-build-isolation
```

TensorRT Python 包必须加 `--no-build-isolation`，否则 pip 构建隔离环境下找不到 `setuptools` 会安装失败。

### 训练

```bash
source venv/bin/activate

# 校验数据集
python scripts/check_dataset.py --root crack_dataset

# 训练 YOLOv8n
python scripts/train.py
```

### 导出 ONNX

```bash
yolo export model=weights/trained/best.pt format=onnx opset=12 simplify=True dynamic=True
```

`dynamic=True` 是后续动态 shape 的前提，`simplify=True` 会清理 ONNX 图中的冗余算子，降低 TensorRT 解析失败的概率。

### 构建 TensorRT Engine

**FP16：**

```bash
cd deploy/trt_python
python onnx_to_engine.py \
  --onnx ../../weights/engines/best.onnx \
  --engine ../../weights/engines/best_fp16.engine \
  --precision fp16
```

**INT8：**

```bash
python onnx_to_int8_engine.py \
  --onnx ../../weights/engines/best.onnx \
  --engine ../../weights/engines/best_int8.engine \
  --calib ../../calib_images \
  --cache ../../calib.cache
```

> Engine 与 GPU 架构、TensorRT 版本强绑定，**不可跨机器复用**。换机器请重新构建。校准缓存 `calib.cache` 可复用，能省掉约 71.7 秒的校准耗时。

### Python 推理测试

```bash
python trt_infer.py \
  --engine ../../weights/engines/best_fp16.engine \
  --image ../../crack_dataset/test/images/00031_jpg.rf.540253f521fc24e940187f74b98ec587.jpg \
  --size 416 \
  --warmup 10 \
  --runs 100
```

**FP16 / INT8 对比：**

```bash
python compare_fp16_int8.py \
  --fp16 ../../weights/engines/best_fp16.engine \
  --int8 ../../weights/engines/best_int8.engine \
  --image-dir ../../crack_dataset/test/images
```

### C++ 推理

```bash
cd deploy/cpp
mkdir -p build && cd build
cmake ..
make -j4

./yolo_trt_demo \
  ../../../weights/engines/best_fp16.engine \
  ../../../crack_dataset/test/images/00031_jpg.rf.540253f521fc24e940187f74b98ec587.jpg \
  output.jpg
```

C++ 侧只依赖 TensorRT + OpenCV + CUDA Runtime，不需要 Python 与 PyTorch，可直接部署到生产环境或 Jetson 设备。

---

## 关键技术点

### 1. 动态 shape 支持

ONNX 导出时使用 `dynamic=True`，TensorRT 构建时配置 optimization profile：

```python
profile.set_shape(
    input_name,
    min=(1, 3, 320, 320),
    opt=(1, 3, 416, 416),
    max=(1, 3, 640, 640),
)
```

实际收益是一个 engine 覆盖 320~640 全部分辨率，无需为每种输入尺寸单独构建引擎。代价是 `max` 设置过大会让 TensorRT 在优化时偏向保守，因此把日常使用的 416 设为 `opt`，让内核自动调优针对该尺寸优化。

### 2. INT8 校准

使用 **Entropy Calibration v2**，200 张校准图，耗时 71.7 秒。

关键发现：YOLOv8 输出头的动态 shape 算子（`Range`、`Shape`、`Cast`）不支持 INT8，TensorRT 会自动把这些层**回退到 FP16**。这导致在 YOLOv8n 这类小模型上，INT8 的延迟收益有限（3.72 → 3.53 ms，约 5%），但仍有明确的工程价值：

- 模型体积减少 **20%**（9.7 MB → 7.8 MB）
- 权重内存减少 **45%**

也就是说 INT8 的价值主要体现在**显存与存储受限的边缘设备**，而非算力受限场景。若想进一步压榨 INT8 性能，需要把输出头的动态算子改写为静态 shape 或自定义 TensorRT plugin。

### 3. C++ 推理核心

```cpp
// 动态 shape：先设 shape 再拿输出维度
context_->setInputShape("images", nvinfer1::Dims4{1, 3, 416, 416});
auto out_dims = context_->getTensorShape("output0");  // [1, 5, 3549]

// 推理
context_->setTensorAddress("images", gpu_input_);
context_->setTensorAddress("output0", gpu_output_);
context_->enqueueV3(stream_);

// 后处理：NMS + DFL 解码
auto detections = postprocess(output_data, img_w, img_h, 0.25f, 0.45f);
```

三个要点：

1. **顺序不能颠倒**：必须先 `setInputShape` 再 `getTensorShape`，否则输出元素数返回 `-5`（无效维度）。
2. **使用 `enqueueV3`**：TensorRT 10.x 的推荐异步接口，配合 CUDA stream 使用，避免隐式同步开销。
3. **手写 DFL 解码**：YOLOv8 是 anchor-free 输出，回归分支是 4×reg_max 的分布，需做 DFL（Distribution Focal Loss）积分还原成坐标，不依赖 Ultralytics 的 Python 后处理。

输出 `[1, 5, 3549]` 中，`5 = 4（框）+ 1（类别）`（单类别），`3549` 是 416 分辨率下三个检测头的锚点总数（52² + 26² + 13² = 2704 + 676 + 169）。

---

## 踩坑记录

| 问题 | 原因 | 解决 |
|---|---|---|
| PyTorch 装不上 | Python 3.14 太新，无对应 wheel | 换成 Python 3.10 |
| C 盘被撑爆 | WSL2 的 vhdx 虚拟磁盘只增不减 | 迁移到 D 盘 + 设置 `TMPDIR` |
| `libcudnn.so.9` 找不到 | ONNX Runtime GPU 版本需要 cuDNN 9 | `pip install nvidia-cudnn-cu12` + 配置 `LD_LIBRARY_PATH` |
| TensorRT 安装失败 | pip 构建隔离环境下找不到 `setuptools` | 加 `--no-build-isolation` |
| `cannot enable executable stack` | glibc 2.41+ 加强了安全限制 | `patchelf --clear-execstack <lib>` |
| `no optimization profile` | 动态 shape 未配置 profile | 添加 `min/opt/max` profile |
| 输出元素数为 `-5` | `getTensorShape` 前未 `setInputShape` | 先调 `setInputShape` |

完整排障记录见 [ENVIRONMENT.md](ENVIRONMENT.md)。

---

## 未来工作

- [ ] C++ 预处理优化（逐像素 → 连续内存访问 / CUDA kernel 化）
- [ ] 多线程流水线（多 engine 并行，提升吞吐）
- [ ] Nsight Systems 性能分析，定位剩余瓶颈
- [ ] Triton Inference Server 部署
- [ ] TensorRT plugin 自定义算子（解决 INT8 回退问题）
- [ ] Jetson 边缘设备实测

---

## 环境详情

| 项目 | 配置 |
|---|---|
| CPU | AMD Ryzen 5 5600H（6 核 12 线程） |
| GPU | NVIDIA RTX 3050 Laptop（4GB） |
| 内存 | 7.7 GB（WSL 分配） |
| 系统 | Windows 11 + WSL2 Ubuntu |
| 驱动 | 595.79 |

---

## 参考

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

---

## 数据集

混凝土裂缝检测数据集，YOLO 格式标注，共 976 张：

| 划分 | 数量 |
|---|---|
| train | 826 |
| valid | 100 |
| test | 50 |

INT8 校准集从训练集中抽取 200 张，与训练/验证集解耦，避免校准偏差。
