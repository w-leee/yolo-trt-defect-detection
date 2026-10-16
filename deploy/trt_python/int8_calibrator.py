"""
TensorRT INT8 校准器
使用 Entropy Calibration v2
"""
import os
import numpy as np
import cv2
import tensorrt as trt
import pycuda.autoinit    # ← 加这行，自动初始化 CUDA context
import pycuda.driver as cuda


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, cache_file, batch_size=1, input_shape=(1, 3, 416, 416)):
        super().__init__()
        self.calib_dir = calib_dir
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape

        self.image_paths = []
        for f in sorted(os.listdir(calib_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                self.image_paths.append(os.path.join(calib_dir, f))
        print(f"[Calib] 校准图片数: {len(self.image_paths)}")

        self.current_idx = 0
        self.device_input = cuda.mem_alloc(int(np.prod(input_shape)) * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx + self.batch_size > len(self.image_paths):
            return None

        batch = np.zeros(self.input_shape, dtype=np.float32)
        for i in range(self.batch_size):
            path = self.image_paths[self.current_idx + i]
            img = cv2.imread(path)
            img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            batch[i] = img

        self.current_idx += self.batch_size
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)