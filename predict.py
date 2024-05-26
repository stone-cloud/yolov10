from ultralytics import YOLOv10
import os
from pathlib import Path
import torch
import json
names = {0: 'car', 1: 'truck', 2: 'bus', 3: 'van', 4: 'freight_car'}
check_point = r'/media/gw/7EF8EB5EF8EB1367/LYH/project/train/weights/yolov10x.pt'
model = YOLOv10(check_point)
save_dir = r'/media/gw/7EF8EB5EF8EB1367/LYH/project/objectDet'
# Run batched inference on a list of images
images = '/media/gw/7EF8EB5EF8EB1367/LYH/project/objectDet/GAIIC2024Y/test/00008.tiff'
results = model(images)  # return a list of Results objects
print(results)
json_str = []
# Process results list
for result in results:
    # stem表示图片不带后缀的名称
    stem = Path(images).stem
    image_id = stem
    # print(int(result.boxes.cls))
    boxes = result.boxes  # Boxes object for bounding box outputs
    if boxes.cls.numel()  == 0:
        print(f'{image_id} No boxes detected')
        json_str.append({
            'image_id': int(image_id),
            'category_id': 'null',
            # round（x,4）表示取x小数点后四位
            'bbox': 'null',
            'score': 'null'
        })
    else:
        for box in boxes:
            json_str.append({
                'image_id': int(image_id),
                'category_id': int(box.cls)+1,
                # round（x,4）表示取x小数点后四位
                'bbox': [round(float(x), 1) for x in box.xywh[0]],
                'score': round(float(box.conf), 4)
            })
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # # obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename='ttt_t.tiff')  # save to disk
json_data = json.dumps(json_str)
# 创建json文件的存放路径，self.save_dir是类中自带的默认存放地址
jsons_path = Path(os.path.join(save_dir, 'json'))
if not os.path.exists(jsons_path):
    os.makedirs(jsons_path)
# 创建json文件，并将self.jdict中的内容写进去
with open(str(jsons_path / f'pred.json'), 'w') as f:
    f.write(json_data)