"""
ONNX → TensorRT engine 转换脚本
支持 FP32 / FP16 / INT8，支持动态 shape
"""
import os
import sys
import time
import argparse
import tensorrt as trt


def build_engine(onnx_path, engine_path, precision="fp16", workspace_gb=2):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX
    print(f"[1/4] 解析 ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX 解析失败：")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False

    # 配置 builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # 动态 shape optimization profile
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"[*] 输入名: {input_name}")

    profile.set_shape(
        input_name,
        min=(1, 3, 320, 320),
        opt=(1, 3, 416, 416),
        max=(1, 3, 640, 640),
    )
    config.add_optimization_profile(profile)
    print("[*] Optimization profile: min(1,3,320,320) opt(1,3,416,416) max(1,3,640,640)")

    # 精度设置
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[2/4] 启用 FP16")
        else:
            print("⚠️  平台不支持 FP16，回退 FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[2/4] 启用 INT8（未校准，仅测试）")
        else:
            print("⚠️  平台不支持 INT8，回退 FP32")
    else:
        print("[2/4] 使用 FP32")

    # 构建 engine
    print("[3/4] 构建 engine（可能需要几分钟）...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("❌ Engine 构建失败")
        return False
    print(f"     构建耗时：{time.time() - t0:.1f} 秒")

    # 保存
    print(f"[4/4] 保存到: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ 完成！engine 大小：{os.path.getsize(engine_path) / 1e6:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="ONNX 文件路径")
    parser.add_argument("--engine", required=True, help="输出 engine 路径")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--workspace", type=int, default=2, help="workspace GB")
    args = parser.parse_args()

    success = build_engine(args.onnx, args.engine, args.precision, args.workspace)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()