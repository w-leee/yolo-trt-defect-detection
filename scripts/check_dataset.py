import os
import sys
import argparse


def check_pair(img_root, label_root, exts=None):
    """检查图片与标签文件是否一一配对"""
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    if not os.path.isdir(img_root):
        print(f"❌ 图片目录不存在：{img_root}")
        return False
    if not os.path.isdir(label_root):
        print(f"❌ 标签目录不存在：{label_root}")
        return False

    img_names = set()
    for f in os.listdir(img_root):
        name, ext = os.path.splitext(f)
        if ext.lower() in exts:
            img_names.add(name)

    label_names = set()
    for f in os.listdir(label_root):
        name, ext = os.path.splitext(f)
        if ext == ".txt":
            label_names.add(name)

    only_img = img_names - label_names
    only_label = label_names - img_names

    print(f"  图片数量：{len(img_names)}")
    print(f"  标签数量：{len(label_names)}")
    print(f"  只有图片没有标签：{only_img if only_img else '无'}")
    print(f"  只有标签没有图片：{only_label if only_label else '无'}")

    return len(only_img) == 0 and len(only_label) == 0


def main():
    parser = argparse.ArgumentParser(description="YOLO 数据集配对校验")
    parser.add_argument("--root", type=str, required=True,
                        help="数据集根目录，例如 crack_dataset")
    args = parser.parse_args()

    root = args.root
    print(f"===== 校验数据集：{root} =====")

    all_ok = True
    for split in ["train", "valid", "test"]:
        print(f"\n--- {split} ---")
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        if os.path.isdir(img_dir):
            ok = check_pair(img_dir, lbl_dir)
            all_ok = all_ok and ok
        else:
            print(f"  跳过（目录不存在）：{img_dir}")

    print(f"\n{'✅ 全部配对成功' if all_ok else '❌ 存在不配对文件'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()