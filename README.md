# 🚀 工业缺陷检测的 TensorRT 边缘推理加速

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorRT](https://img.shields.io/badge/TensorRT-10.7-green)
![CUDA](https://img.shields.io/badge/CUDA-12.6-orange)
![C++](https://img.shields.io/badge/C++--17-red)

基于 **YOLOv8 + ONNX + TensorRT** 的工业缺陷实时检测与边缘推理优化。以**混凝土裂缝（Crack）**为典型缺陷样本，完整打通从模型训练、ONNX 导出、TensorRT FP16/INT8 量化，到 C++ 高性能推理的全链路部署。

在 RTX 3050 Laptop（4GB 显存）这类典型边缘算力上，C++ 端到端（预处理 + 推理 + 后处理）耗时压至 **4.45 ms**，实测峰值突破 **224.5 FPS**，充分满足了工业在线检测的严苛实时性要求。

---

## 📖 目录

- [项目亮点](#-项目亮点)
- [性能与精度基准](#-性能与精度基准)
- [技术架构与数据流](#️️-技术架构与数据流)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [硬核技术点解析](#-硬核技术点解析)
- [踩坑记录 (Debug Log)](#️-踩坑记录-debug-log)
- [未来工作](#-未来工作)

---

## 🌟 项目亮点

- **全链路部署闭环**：PyTorch → ONNX → TensorRT FP16/INT8 → C++ 推理，每一环都有可复现脚本与实测数据。
- **极限端到端性能**：C++ 端到端（预处理+推理+手写后处理）仅需 **4.45 ms，峰值 224 FPS**。
- **INT8 量化深度探索**：模型体积减少 20%，权重内存减少 45%，并精准归因了动态算子回退 FP16 的底层原因。
- **手写 NMS + DFL 解码**：完全剥离 Ultralytics 后处理，C++ 侧实现零 Python、零 PyTorch 依赖。
- **动态 Shape 支持**：单引擎覆盖 320×320 ~ 640×640 多分辨率输入，适配多场景相机。

---

## 📊 性能与精度基准

### 性能压测（RTX 3050 Laptop, 416×416, warmup=10, runs=100）

| 阶段 | 延迟 (ms) | FPS | 模型大小 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Python)** | 14.6 ms | 68 | 6.2 MB | 纯推理（Baseline） |
| **ONNX Runtime (Python)** | ~10 ms | ~100 | 12.5 MB | 未单独精细测量 |
| **TensorRT FP16 (Python)** | 3.72 ms | 269 | 9.7 MB | 纯推理 |
| **TensorRT INT8 (Python)** | 3.53 ms | 221 | 7.8 MB | 纯推理 |
| **🔥 TensorRT FP16 (C++ 端到端)** | **4.45 ms** | **224** | - | **预处理 + 推理 + 手写后处理** |

> **结论**：FP16 引擎纯推理相比 PyTorch 提速约 **3.9×**；C++ 端到端耗时控制在 5ms 以内，说明预处理与手写后处理不再是瓶颈，整条管线处于计算受限的最优区间。

### 模型精度（YOLOv8n, 单类别）

| 指标 | 数值 | 说明 |
| :--- | :--- | :--- |
| **mAP50** | 0.847 | - |
| **mAP50-95** | 0.825 | - |
| **Precision** | 0.661 | - |
| **Recall** | **0.942** | **业务取舍：工业漏检代价远高于误检，刻意容忍误报换取高召回**。 |

---

## 🏗️ 技术架构与数据流

```text
┌─────────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────────┐
│  PyTorch    │───▶│  ONNX   │───▶│  TensorRT    │───▶│  C++ +       │
│  YOLOv8n    │    │ 12.5 MB │    │  FP16 / INT8 │    │  OpenCV      │
└─────────────┘    └─────────┘    └──────────────┘    └──────────────┘
    14.6 ms          ~10 ms           3.72 ms           4.45 ms (224 FPS)
```

**C++ 端数据流转**：

图像输入 → letterbox 缩放至 416×416 → 归一化 / HWC→CHW → GPU memcpy → TensorRT enqueueV3 异步推理 → 输出 [1, 5, 3549] → 手写 DFL 解码 + 框还原 → 手写 NMS → OpenCV 绘制结果。

---

## 📂 项目结构

> **注**：为了保证仓库轻量化，项目已配置严格的 `.gitignore`。大型文件（如数据集、预训练权重、第三方 SDK、编译产物等）不会提交到 Git。克隆本项目后，需参照下方说明自行准备。

```text
yolo-demo/
├── .vscode/                           # VS Code 配置 (部分保留)
│   ├── c_cpp_properties.json          # [Tracked] 提供 C++ 头文件路径与环境感知
│   └── extensions.json                # [Tracked] 推荐插件列表
├── calib_images/                      # [Git Ignored] INT8 量化校准集 (200 张)
├── crack_dataset/                     # [Git Ignored] 混凝土裂缝数据集 (YOLO 格式)
├── deploy/                            # 🚀 核心部署代码区 (主要维护区域)
│   ├── cpp/                           # C++ TensorRT 高性能推理工程
│   │   ├── build/                     # [Git Ignored] CMake 构建中间产物
│   │   ├── include/                   # C++ 头文件目录
│   │   │   └── trt_engine.h           # [Tracked] TensorRT 引擎封装类声明
│   │   ├── src/                       # C++ 源码目录
│   │   │   ├── main.cpp               # [Tracked] 主程序入口 (命令行参数解析)
│   │   │   └── trt_engine.cpp         # [Tracked] 核心实现 (预处理/推理/NMS/DFL)
│   │   └── CMakeLists.txt             # [Tracked] C++ 工程构建脚本
│   ├── trt_python/                    # Python 端量化与验证脚本
│   │   ├── int8_calibrator.py         # [Tracked] INT8 熵校准器实现
│   │   ├── onnx_to_engine.py          # [Tracked] ONNX 转 FP16 Engine 脚本
│   │   ├── onnx_to_int8_engine.py     # [Tracked] ONNX 转 INT8 Engine 脚本
│   │   └── trt_infer.py               # [Tracked] Python 端推理与性能基准测试
│   ├── docs/                          # 性能报告与文档说明
│   └── engine/                        # [Git Ignored] 存放部署用的 .engine 副本
├── runs/                              # [Git Ignored] YOLOv8 训练日志与权重
├── scripts/                           # Python 辅助脚本
│   ├── check_dataset.py               # [Tracked] 数据集完整性校验脚本
│   └── train.py                       # [Tracked] YOLOv8n 训练启动脚本
├── third_party/                       # [Git Ignored] TensorRT C++ SDK 本地依赖库
├── venv/                              # [Git Ignored] Python 虚拟环境
├── weights/                           # [Git Ignored] 模型权重目录
│   ├── pretrained/                    # 官方预训练权重 (.pt)
│   ├── trained/                       # 训练产出的 best.pt
│   └── engines/                       # 导出的 ONNX 与 TRT Engine (.engine)
├── .gitignore                         # Git 忽略规则
├── ENVIRONMENT.md                     # [Tracked] 双系统开发环境基准档案
├── LICENSE                            # [Tracked] 开源协议 (MIT)
└── README.md                          # [Tracked] 项目说明文档
```

---

## 🚀 快速开始

### 1. 环境要求

| 组件 | 版本 |
| :--- | :--- |
| 系统 | WSL2 Ubuntu 26.04（原生 Linux 也可） |
| CUDA / TensorRT | CUDA 12.6 / TensorRT 10.7 |
| Python | 3.10（强烈建议，3.14 太新暂无 PyTorch Wheel） |
| C++ | GCC 15.2 / CMake 4.2 |

> 注：完整双系统环境配置请参阅 `ENVIRONMENT.md`

### 2. 安装与训练

```bash
# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装核心依赖
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics onnx onnxsim onnxruntime-gpu opencv-python
pip install tensorrt==10.7 --no-build-isolation

# 训练与导出 (需提前准备好数据集)
python scripts/train.py
yolo export model=weights/trained/best.pt format=onnx opset=12 simplify=True dynamic=True
```

### 3. C++ 编译与极限测试

```bash
cd deploy/cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 执行端到端推理测试 (416x416)
./yolo_trt_demo ../../weights/engines/best_fp16.engine ../../crack_dataset/test/images/xxx.jpg output.jpg
```

---

## 🧠 硬核技术点解析

### 1. 动态 Shape 支持

ONNX 导出时使用 `dynamic=True`，TensorRT 构建时配置 optimization profile（`min=320, opt=416, max=640`），实现单引擎覆盖所有分辨率。且 `opt` 设为 416，让内核自动调优针对该常用尺寸极致优化。

### 2. INT8 量化的"玄学"与真相

使用 **Entropy Calibration v2** 耗时 71.7 秒完成校准。

**关键发现**：YOLOv8 输出头的动态 shape 算子（Range、Shape、Cast）不支持 INT8，TensorRT 会自动将其回退到 FP16。这导致小模型（YOLOv8n）在 INT8 下延迟收益有限（3.72 → 3.53 ms），但模型体积减少 20%，权重内存减少 45%。**INT8 的价值在显存受限的边缘设备上巨大，若要压榨极限性能，需改写静态 Shape 或手写 TRT Plugin。**

### 3. C++ 核心推理与手写后处理

- **顺序铁律**：必须先 `context->setInputShape()` 再 `context->getTensorShape()`，否则输出元素数会返回 -5（无效维度）。
- **异步推演**：使用 TensorRT 10.x 推荐的 `enqueueV3`，配合 CUDA stream 消除隐式同步开销。
- **手写 DFL 解码**：YOLOv8 是 anchor-free 输出，回归分支为 4×reg_max 分布，C++ 侧完全手写 DFL 积分还原坐标与 NMS，彻底摆脱对 Python 运行时的依赖。

---

## 🕳️ 踩坑记录 (Debug Log)

| 问题现象 | 根因分析 | 解决方案 |
| :--- | :--- | :--- |
| PyTorch 装不上 | Python 3.14 太新，官方生态无对应 wheel | 降级并使用 Python 3.10 虚拟环境 |
| C 盘被撑爆 | WSL2 的 vhdx 虚拟磁盘只增不减 | 将 WSL2 整个迁移到 D 盘，并设置 TMPDIR |
| 找不到 libcudnn.so.9 | ONNX Runtime GPU 依赖 cuDNN 9 | `pip install nvidia-cudnn-cu12` 并配置 LD_LIBRARY_PATH |
| TensorRT 安装失败 | pip 构建隔离环境下找不到 setuptools | 必须加上 `--no-build-isolation` 参数 |
| getTensorShape 返回 -5 | 动态 shape 下未先 setInputShape | 严格保证调用顺序：先 set 再 get |
| C++ 编译找不到 NvInfer.h | pip 安装的包不包含 C++ 开发头文件 | 从官方下载完整 tar.gz，将 include 和 lib 软链接到 /usr/local/ |
| 可执行栈限制 | glibc 2.41+ 安全机制加强 | `patchelf --clear-execstack <lib_path>` |

---

## 🔮 未来工作

- [ ] **C++ 预处理 CUDA 化**：将逐像素的 CPU 预处理迁移到 CUDA Kernel，实现全链路零内存拷贝。
- [ ] **多线程流水线**：多 Engine 并行，提升工业产线的高并发吞吐量。
- [ ] **Nsight Systems 调优**：深度定位剩余的 CPU/GPU 异步瓶颈。
- [ ] **Triton Inference Server 部署**：接入云端或边缘节点统一调度。
- [ ] **Jetson Orin 实测**：在真实工业边缘硬件上进行功耗与算力压测。

---

**Author**: w-leee | **Dataset**: 混凝土裂缝检测数据集 (YOLO格式, 共976张) | **License**: MIT
