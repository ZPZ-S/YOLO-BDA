import os
import json
from tqdm import tqdm

def convert_yolo_to_coco(yolo_dataset_dir, coco_output_dir, classes_file):
    """
    Convert YOLO format dataset to COCO format.

    Args:
        yolo_dataset_dir (str): YOLO dataset root directory.
        coco_output_dir (str): Output directory for COCO JSON files.
        classes_file (str): Path to the file containing class names (one per line).
    """
    # Load class names
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(classes)]

    # YOLO subdirectories
    splits = ['train', 'valid', 'test']
    for split in splits:
        image_dir = os.path.join(yolo_dataset_dir, 'images', split)
        label_dir = os.path.join(yolo_dataset_dir, 'labels', split)

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        annotation_id = 1

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Skipping {split} as it does not exist.")
            continue

        # Process images and labels
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        for image_id, image_file in enumerate(tqdm(image_files), 1):
            # Add image info
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            if not os.path.exists(label_path):
                print(f"Warning: Label file for {image_file} not found, skipping.")
                continue

            # Assume all images are 1024x1024, or replace with actual size
            width, height = 640, 640
            coco_data["images"].append({
                "id": image_id,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height
            })

            # Add annotations
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height
                    x_min = x_center - box_width / 2
                    y_min = y_center - box_height / 2

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id) + 1,  # YOLO class index starts at 0, COCO starts at 1
                        "bbox": [x_min, y_min, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        # Save COCO JSON
        os.makedirs(coco_output_dir, exist_ok=True)
        coco_output_path = os.path.join(coco_output_dir, f"{split}.json")
        with open(coco_output_path, 'w') as f:
            json.dump(coco_data, f, indent=4)

        print(f"Saved {split} COCO annotations to {coco_output_path}")


# 使用示例
yolo_dataset_dir = '/root/autodl-tmp/ultralytics-yolo11-main/dataset/56-min-1560'  # 数据集根目录
coco_output_dir = '/root/autodl-tmp/ultralytics-yolo11-main/dataset/coco-56-min-1560'  # 输出的COCO格式目录
classes_file = '/root/autodl-tmp/ultralytics-yolo11-main/dataset/56-min-1560/classes.txt'  # 包含类别名称的文件

convert_yolo_to_coco(yolo_dataset_dir, coco_output_dir, classes_file)
