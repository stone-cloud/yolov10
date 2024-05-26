import json
import os
import shutil
from tqdm import tqdm

coco_path = "/media/gw/7EF8EB5EF8EB1367/LYH/project/objectDet/GAIIC2024Y"
output_path = "/media/gw/7EF8EB5EF8EB1367/LYH/project/objectDet/GAIIC2024y"

os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)

with open(os.path.join(coco_path, "train", "train.json"), "r") as f:
    train_annotations = json.load(f)

with open(os.path.join(coco_path, "val", "val.json"), "r") as f:
    val_annotations = json.load(f)

for image in tqdm(val_annotations["images"]):
    width, height = image["width"], image["height"]
    scale_x = 1.0 / width
    scale_y = 1.0 / height

    label = ""
    for annotation in val_annotations["annotations"]:
        if annotation["image_id"] == image["id"]:
            # Convert the annotation to YOLO format
            x, y, w, h = annotation["bbox"]
            x_center = x + w / 2.0
            y_center = y + h / 2.0
            x_center *= scale_x
            y_center *= scale_y
            w *= scale_x
            h *= scale_y
            class_id = annotation["category_id"]
            if class_id == 0:
                print(class_id)
            label += "{} {} {} {} {}\n".format(class_id, x_center, y_center, w, h)
