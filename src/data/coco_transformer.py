import json


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
