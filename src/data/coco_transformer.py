import json
from typing import Any


class COCOjsonTransformer:
    def __init__(self, coco: str | dict, 
                 instructions: dict = {},
                 min_width: float = 40, min_height: float = 40):
        """
        Transforms a coco.json file into a new coco.json file from a
        instructions dictionary.
        """

        self.coco = self._coco(coco)
        self.class_names, self.class_names_inv, self.class_id_mapper = self._make_class_names(self.coco)
        self._transform(instructions, min_width, min_height)
    

    @staticmethod
    def _coco(coco):
        if isinstance(coco, str):
            with open(coco, 'r') as oj:
                coco = json.load(oj)
        elif isinstance(coco, dict):
            coco = coco 
        else:
            print("annotations wernt loaded")
        return coco 

    
    @staticmethod
    def _make_class_names(coco):
        class_names = {}
        for annote in coco['annotations']:
            cat_id  = annote['category_id']

            for catinfo in coco['categories']:
                if catinfo['id'] == cat_id:
                    if cat_id not in class_names.keys():
                        class_names[cat_id] = catinfo['name']

        class_names_inv = {}
        for id, cat in class_names.items():
            if cat not in class_names_inv:
                class_names_inv[cat] = [id]
            else:
                class_names_inv[cat].append(id)

        class_id_mapper = {
            class_id: i for i, class_id in enumerate(class_names.keys())
        }

        return class_names, class_names_inv, class_id_mapper


    def _transform(self, instructions, min_width, min_heght):
        new_coco = {'images': [], 'annotations': [], 'categories': []}
        image_ids = []
        for annote in self.coco['annotations']:
            annote_cat = self.class_names[annote['category_id']]
            if annote_cat in instructions.keys():
                if instructions[annote_cat] == 'ignore':
                    continue
            elif annote['bbox'][2] < min_width or annote['bbox'][3] < min_heght:
                continue
            else:
                new_coco['annotations'].append(annote)
                image_ids.append(annote['image_id'])

        for image in self.coco['images']:
            if image['id'] not in image_ids:
                continue
            else:
                new_coco['images'].append(image)

        for category in self.coco['categories']:
            if category['name'] in instructions.keys():
                if instructions[category['name']] == 'ignore':
                    continue
                else:
                    new_coco['categories'].append(
                        {
                            'id': category['id'], 
                            'name': instructions[category['name']], 
                        }
                    )
            else:
                new_coco['categories'].append(category)
        
        self.instructions  = instructions
        self.coco = new_coco
        self.class_names, self.class_names_inv, self.class_id_mapper = self._make_class_names(self.coco)

        return None


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

    assert type(coco) == dict[str, list[dict[str, Any]]]

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
