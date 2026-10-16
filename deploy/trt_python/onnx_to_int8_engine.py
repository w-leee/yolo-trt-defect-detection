"""
ONNX → TensorRT INT8 engine（PTQ + 校准）
"""
import os
import sys
import time
import argparse
import tensorrt as trt
from int8_calibrator import Int8Calibrator


def build_int8_engine(onnx_path, engine_path, calib_dir, cache_file, workspace_gb=2):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"[1/5] 解析 ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        min=(1, 3, 320, 320),
        opt=(1, 3, 416, 416),
        max=(1, 3, 640, 640),
    )
    config.add_optimization_profile(profile)
    print(f"[*] Input: {input_name}")

    if not builder.platform_has_fast_int8:
        print("❌ 平台不支持 INT8")
        return False
    config.set_flag(trt.BuilderFlag.INT8)

    calibrator = Int8Calibrator(calib_dir, cache_file, batch_size=1, input_shape=(1, 3, 416, 416))
    config.int8_calibrator = calibrator
    print(f"[2/5] INT8 校准器已设置，缓存: {cache_file}")

    print("[3/5] 构建 INT8 engine（含校准，可能 5~10 分钟）...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("❌ 构建失败")
        return False
    print(f"     耗时: {time.time() - t0:.1f} 秒")

    print(f"[4/5] 保存: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[5/5] ✅ 完成！大小: {os.path.getsize(engine_path) / 1e6:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--calib", required=True, help="校准集目录")
    parser.add_argument("--cache", default="calib.cache")
    parser.add_argument("--workspace", type=int, default=2)
    args = parser.parse_args()

    ok = build_int8_engine(args.onnx, args.engine, args.calib, args.cache, args.workspace)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()