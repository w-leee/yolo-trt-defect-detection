from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = "/home/wangle/workspace/yolo_trt/yolo-demo"


def main():
    # 加载 YOLOv8n 官方预训练权重
    model = YOLO(f"{PROJECT_ROOT}/weights/yolov8n.pt")

    model.train(
        data=f"{PROJECT_ROOT}/crack_dataset/data.yaml",
        epochs=150,
        imgsz=416,
        batch=8,
        amp=True,

        # 数据增强（小数据集必备）
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        degrees=10,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,

        # 训练策略
        patience=50,
        lr0=0.001,
        warmup_epochs=5,

        # 输出路径
        project=f"{PROJECT_ROOT}/runs/detect",
        name="train",
        exist_ok=True,   # 允许覆盖同名目录
    )


if __name__ == "__main__":
    main()