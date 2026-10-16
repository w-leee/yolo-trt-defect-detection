"""
对比 FP16 和 INT8 engine 的输出差异
"""
import os
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTEngine:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        self.stream = cuda.Stream()

    def infer(self, input_np):
        self.context.set_input_shape(self.input_name, input_np.shape)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_np = np.empty(output_shape, dtype=np.float32)
        d_in = cuda.mem_alloc(input_np.nbytes)
        d_out = cuda.mem_alloc(output_np.nbytes)
        self.context.set_tensor_address(self.input_name, int(d_in))
        self.context.set_tensor_address(self.output_name, int(d_out))
        cuda.memcpy_htod_async(d_in, input_np, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_np, d_out, self.stream)
        self.stream.synchronize()
        d_in.free()
        d_out.free()
        return output_np


def preprocess(img_path, size=416):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None, ...])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--size", type=int, default=416)
    args = parser.parse_args()

    fp16_engine = TRTEngine(args.fp16)
    int8_engine = TRTEngine(args.int8)

    imgs = [f for f in sorted(os.listdir(args.images))
            if f.lower().endswith((".jpg", ".png", ".jpeg"))][:args.num]

    print(f"对比 {len(imgs)} 张图\n")
    print(f"{'图片':<40} {'FP16 max':<12} {'INT8 max':<12} {'max diff':<12} {'相对误差'}")
    print("-" * 90)

    max_diffs = []
    for name in imgs:
        path = os.path.join(args.images, name)
        inp = preprocess(path, args.size)

        out_fp16 = fp16_engine.infer(inp)
        out_int8 = int8_engine.infer(inp)

        diff = np.abs(out_fp16 - out_int8)
        max_diff = diff.max()
        rel_err = max_diff / (np.abs(out_fp16).max() + 1e-6)

        max_diffs.append(max_diff)
        print(f"{name[:38]:<40} {out_fp16.max():<12.2f} {out_int8.max():<12.2f} {max_diff:<12.2f} {rel_err*100:.2f}%")

    print()
    print("=" * 60)
    print(f"平均最大差值: {np.mean(max_diffs):.4f}")
    print(f"最大差值:     {np.max(max_diffs):.4f}")
    print(f"结论: 差值 < 1.0 认为精度保持良好")


if __name__ == "__main__":
    main()