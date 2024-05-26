from ultralytics import YOLOv10
import os.path as osp
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

check_point = r'/media/gw/7EF8EB5EF8EB1367/LYH/project/train/weights/yolov10x.pt'
model = YOLOv10(check_point)

# Run batched inference on a list of images
root = r'/media/gw/7EF8EB5EF8EB1367/LYH/project/objectDet/GAIIC2024Y/test'
file_list = os.listdir(root)
file_list.sort()
# images = [osp.join(root, file) for file in file_list]
save_dir = r'/media/gw/7EF8EB5EF8EB1367/LYH/project/objectDet'
json_str = []
pred_root = osp.join(save_dir, 'pred/project')
if not os.path.exists(pred_root):
    os.makedirs(pred_root)
# Process results list
for file in tqdm(file_list):
    image = osp.join(root, file)
    # stem表示图片不带后缀的名称
    stem = Path(image).stem
    image_id = stem
    print(image)
    result = model(image)[0]  # return a list of Results objects
    print(result)
    boxes = result.boxes  # Boxes object for bounding box outputs
    if boxes.cls.numel()  == 0:
        print(f'{file} No boxes detected')
        json_str.append({
            'image_id': int(image_id),
            'category_id': None,
            # round（x,4）表示取x小数点后四位
            'bbox': None,
            'score': None
        })
    else:
        for box in boxes:
            json_str.append({
                'image_id': int(image_id),
                'category_id': int(box.cls) + 1,
                # round（x,4）表示取x小数点后四位
                'bbox': [round(float(x), 1) for x in box.xywh[0]],
                'score': round(float(box.conf), 4)
            })
    # result.save(filename=osp.join(pred_root, file))  # save to disk
json_data = json.dumps(json_str)
# 创建json文件的存放路径，self.save_dir是类中自带的默认存放地址
jsons_path = Path(os.path.join(save_dir, 'json'))
if not os.path.exists(jsons_path):
    os.makedirs(jsons_path)
# 创建json文件，并将self.jdict中的内容写进去
with open(str(jsons_path / f'pred.json'), 'w') as f:
    f.write(json_data)