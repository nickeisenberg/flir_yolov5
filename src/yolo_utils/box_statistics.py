import numpy as np
import json


def make_coco_stats(coco: dict | str, pop_nil=True):

    if isinstance(coco, str):
        with open(coco, "r") as oj:
            coco = json.load(oj)
    
    assert type(coco) == dict

    id_to_name = {
        cat["id"]: cat["name"] 
        for cat in coco["categories"]
    }

    name_to_id = {
        cat["name"]: cat["id"] 
        for cat in coco["categories"]
    }
    
    cat_stats = {
        cat: {"count": 0, "bboxes": []} 
        for cat in name_to_id 
    }
    
    for ann in coco["annotations"]:
        cat = id_to_name[ann["category_id"]]
        bbox = ann["bbox"]
        cat_stats[cat]["count"] += 1
        cat_stats[cat]["bboxes"].append(bbox)
    
    for n in cat_stats:
        cat_stats[n]["bboxes"] = np.array(cat_stats[n]["bboxes"])
    
    count_tups = sorted({n: cat_stats[n]["count"] for n in cat_stats}.items(), key=lambda x: x[1])
    if pop_nil:
        for tup in count_tups[:: -1]:
            if tup[1] == 0:
                _ = cat_stats.pop(tup[0])

    return cat_stats
