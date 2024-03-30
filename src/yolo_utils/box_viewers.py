import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import os


def view_boxes_from_coco_image_id(coco: str | dict, 
                                  image_id: int, 
                                  file_name_root: str, 
                                  show=True):
    if isinstance(coco, str):
        try:
            with open(coco, "r") as oj:
                coco = json.load(oj)
        except Exception as e:
            raise e

    assert type(coco) == dict

    boxes = [
        ann["bbox"]
        for ann in coco["annotations"] if ann["image_id"] == image_id
    ]
    
    file_name = None
    for img in coco["images"]:
        if img["id"] == image_id:
            file_name = img["file_name"]
            break
    assert file_name is not None
    
    file_name = os.path.join(file_name_root, file_name)

    fig = view_boxes(file_name, boxes, show=False)

    if show:
        plt.show()
    else:
        return fig


def view_boxes(img: str | Image.Image, boxes: list, show=True):
    if isinstance(img, str):
        img = Image.open(img)
    
    if img.mode == 'L':
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        img = rgb_img
    
    draw = ImageDraw.Draw(img)
    
    for bbox in boxes:
        x0, y0, w, h = bbox
        x1, y1 = x0 + w, y0 + h
        draw.rectangle((x0, y0, x1, y1), outline ="red")

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    
    fig = plt.imshow(torch.permute(transform(img), (1, 2, 0)))

    if show:
        plt.show()

    else:
        return fig


if __name__ == "__main__":
    home = os.environ["HOME"]
    root = f"{home}/Datasets/flir/images_thermal_train"
    with open(os.path.join(root, "coco.json"), "r") as oj:
        coco = json.load(oj)
    
    view_boxes_from_coco_image_id(coco, 11, root)
    
