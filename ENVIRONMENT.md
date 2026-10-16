# 环境配置

> 最后更新：2026-10-16

## 硬件
- CPU: AMD Ryzen 5 5600H (6 核 12 线程)
- GPU: NVIDIA RTX 3050 Laptop (4GB 显存)
- 内存: 7.7 GB (WSL 分配)
- 硬盘: SK Hynix 512G SSD (C/D 同盘，WSL 已迁移到 D 盘)

## 系统
- Windows 11（宿主）
- WSL2 Ubuntu（已迁移到 `D:\WSL\Ubuntu`）
- NVIDIA 驱动 595.79（Windows 端）
- Docker Desktop（支持 `--gpus all`）

## 软件版本
| 组件 | 版本 |
|---|---|
| CUDA Toolkit | 12.6 |
| cuDNN | 9.x（pip 装 nvidia-cudnn-cu12） |
| Python | 3.10.22 |
| PyTorch | 2.5.1+cu121 |
| TorchVision | 0.20.1+cu121 |
| Ultralytics | 8.4.177 |
| ONNX | 1.23.2 |
| ONNX Runtime GPU | 1.31.0 |
| ONNXSIM | 0.8.1 |
| onnxslim | 0.1.98 |
| TensorRT | 10.7.0（tensorrt-cu12） |
| pycuda | （待装） |

## 关键路径
- 项目根目录: `/home/wangle/workspace/yolo_trt/yolo-demo`
- 虚拟环境: `/home/wangle/workspace/yolo_trt/yolo-demo/venv`
- 模型权重: `weights/best.pt`（mAP50=0.847, mAP50-95=0.825）
- ONNX 文件: `weights/best.onnx`（12.5 MB，动态 shape）
- TensorRT engine: `weights/best_fp16.engine`（10.2 MB）
- 数据集: `crack_dataset/`（train 826 / valid 100 / test 50）
- 训练输出: `runs/detect/train/`

## 环境变量（已写入 ~/.bashrc）
```bash
export LD_LIBRARY_PATH=/home/wangle/workspace/yolo_trt/yolo-demo/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export TMPDIR=/home/wangle/tmp

激活环境
cd ~/workspace/yolo_trt/yolo-demo
source venv/bin/activate

常用命令
# 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"

# 验证 TensorRT
python -c "import tensorrt as trt; b = trt.Builder(trt.Logger(trt.Logger.WARNING)); print('OK')"

# 验证 ONNX Runtime Provider
python -c "import onnxruntime as ort; print(ort.get_available_providers())"