import json

def coco_transformer(coco: str | dict,
                     class_instructions: dict[str, str] | None = None,
                     x_min_max_width: tuple[int, int] | None = None,
                     y_min_max_width: tuple[int, int] | None = None,
                     x_pad: tuple[int, int] | None = None,
                     y_pad: tuple[int, int] | None = None):

    
    if isinstance(coco, str):
        try:
            with open(coco, "r") as oj:
                coco = json.load(oj)
        except Exception as e:
            raise e

    assert type(coco) == dict

    class_name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    class_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    transformed_class_name_to_id = {}
    image_ids_to_keep = []
    transformed_annots = []
    for annot in coco["annotations"]:
        
        x0, y0, w, h = annot["bbox"]
        x1, y1 = x0 + w, y0 + h

        if x_min_max_width:
            if w < x_min_max_width[0] or w > x_min_max_width[1]:
                continue

        if y_min_max_width:
            if h < y_min_max_width[0] or h > y_min_max_width[1]:
                continue

        if x_pad:
            if x0 < x_pad[0] or x1 > x_pad[1]:
                continue

        if y_pad:
            if y0 < y_pad[0] or y1 > y_pad[1]:
                continue

        cat_name = class_id_to_name[annot["category_id"]]

        if class_instructions:
            if cat_name in class_instructions:
                if class_instructions[cat_name] == "ignore":
                    continue

                else:
                    new_cat_name = class_instructions[cat_name]
                    if not new_cat_name in transformed_class_name_to_id:
                        new_id = len(class_name_to_id) 
                        new_id += len(transformed_class_name_to_id)
                        transformed_class_name_to_id[new_cat_name] = new_id
                       
            else:
                new_cat_name = cat_name
                transformed_class_name_to_id[cat_name] = class_name_to_id[cat_name]

        else:
            new_cat_name = cat_name
            transformed_class_name_to_id = class_name_to_id
                    
        transformed_annots.append(
            {
                "bbox": [x0, y0, w, h],
                "image_id": annot["image_id"],
                "id": len(transformed_annots),
                "category_id": transformed_class_name_to_id[new_cat_name]
            }
        )

        if annot["image_id"] not in image_ids_to_keep:
            image_ids_to_keep.append(annot["image_id"])

    transformed_images = [
        {"file_name": img["file_name"], "id": img["id"]}
        for img in coco["images"] if img["id"] in image_ids_to_keep
    ]

    tranformed_categories = [
        {"name": name, "id": id} 
        for name, id in transformed_class_name_to_id.items()
    ]

    transformed_coco = {
        'images': transformed_images, 
        'annotations': transformed_annots, 
        'categories': tranformed_categories
    }

    return transformed_coco

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(f"{os.environ['HOME']}GitRepos/flir_yolov5")
    from src.utils.box_viewers import view_boxes_from_coco_image_id
    
    home = os.environ["HOME"]
    with open(f"{home}/Datasets/flir/images_thermal_train/coco.json", "r") as oj:
        coco = json.load(oj)
    
    
    class_instructions = {}
    for cat in coco["categories"]:
        if cat["name"] in ["person", "car"]:
            continue
        elif cat["name"] in ["bus", "truck"]:
            class_instructions[cat["name"]] = "bus_and_truck"
        else:
            class_instructions[cat["name"]] = "ignore"
    
    transformed_coco = coco_transformer(
        coco,
        class_instructions=class_instructions,
        x_min_max_width=(10, 612),
        y_min_max_width=(10, 540),
        x_pad=(10, 602),
        y_pad=(1, 530)
    )
    
    print(transformed_coco["categories"])
    
    for x in transformed_coco["annotations"]:
        if x["category_id"] == 82:
            print(x)
            break
    
    root = f"{home}/Datasets/flir/images_thermal_train"
    
    view_boxes_from_coco_image_id(transformed_coco, 9, root)
    
    view_boxes_from_coco_image_id(coco, 9, root)

