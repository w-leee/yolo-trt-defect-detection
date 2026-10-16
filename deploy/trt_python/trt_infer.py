"""
TensorRT engine 推理 + 性能测试
"""
import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 找输入输出名
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        print(f"[*] Input:  {self.input_name}")
        print(f"[*] Output: {self.output_name}")

        self.stream = cuda.Stream()

    def infer(self, input_np):
        """input_np: (1,3,H,W) float32"""
        # 设定输入 shape
        self.context.set_input_shape(self.input_name, input_np.shape)

        # 获取输出 shape（在设定输入后才知道）
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_np = np.empty(output_shape, dtype=np.float32)

        # 分配 device 内存
        d_input = cuda.mem_alloc(input_np.nbytes)
        d_output = cuda.mem_alloc(output_np.nbytes)

        # 绑定地址
        self.context.set_tensor_address(self.input_name, int(d_input))
        self.context.set_tensor_address(self.output_name, int(d_output))

        # 拷贝输入 → 执行 → 拷回输出
        cuda.memcpy_htod_async(d_input, input_np, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_np, d_output, self.stream)
        self.stream.synchronize()

        # 释放显存
        d_input.free()
        d_output.free()

        return output_np


def preprocess(img_path, size=416):
    """读图 → resize → RGB → 归一化 → CHW → batch"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"读不到图片: {img_path}")
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)      # HWC → CHW
    img = np.expand_dims(img, 0)       # → (1,3,H,W)
    return np.ascontiguousarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    print(f"[1] 加载 engine: {args.engine}")
    engine = TRTEngine(args.engine)

    print(f"[2] 预处理图片: {args.image}")
    input_np = preprocess(args.image, args.size)
    print(f"    输入 shape: {input_np.shape}")

    print(f"[3] Warmup {args.warmup} 次...")
    for _ in range(args.warmup):
        engine.infer(input_np)

    print(f"[4] 测速 {args.runs} 次...")
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = engine.infer(input_np)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print()
    print("=" * 40)
    print(f"平均延迟: {times.mean():.2f} ms")
    print(f"最小延迟: {times.min():.2f} ms")
    print(f"最大延迟: {times.max():.2f} ms")
    print(f"P50:      {np.percentile(times, 50):.2f} ms")
    print(f"P99:      {np.percentile(times, 99):.2f} ms")
    print(f"FPS:      {1000 / times.mean():.1f}")
    print("=" * 40)
    print(f"输出 shape: {out.shape}")
    print(f"输出最小值: {out.min():.4f}, 最大值: {out.max():.4f}")


if __name__ == "__main__":
    main()