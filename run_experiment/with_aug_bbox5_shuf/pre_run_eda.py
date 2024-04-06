import os
import json
import numpy as np
from src.yolo_utils.coco_transformer import coco_transformer
from src.yolo_utils.box_statistics import make_coco_stats

flir_root = os.path.expanduser("~/datasets/flir")

path = os.path.join(flir_root, "images_thermal_train", "coco.json")
with open(path, "r") as oj:
    coco = json.load(oj)

instructions = {}
for cat in coco["categories"]:
    name = cat["name"]
    if name not in [
        'car', 
        'person', 
        'bike', 
        'bus', 
        'other vehicle',
        'motor', 
        'truck', 
    ]:
        instructions[name] = "ignore"

tcoco = coco_transformer(
    coco, instructions, (25, 640), (25, 512), (0, 640), (0, 512)
)


stats = make_coco_stats(coco)

counts = sorted({k: stats[k]["count"] for k in stats}.items(), key=lambda x: x[1])[::-1]

for tup in counts:
    print(tup)


for cat in stats:
    bbox = stats[cat]["bboxes"]
    print(cat, np.quantile(bbox[:, 2:], [.30], axis=0))
    print(cat, np.mean(bbox[:, 2:], axis=0))
    print(cat, np.min(bbox[:, 2:], axis=0))
