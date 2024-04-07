from copy import deepcopy
import json
from PIL import Image, ImageDraw
from PIL.ImageFont import ImageFont
import matplotlib.pyplot as plt
import torch
from torch import Tensor
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


def view_pred_vs_actual(img: Image.Image, 
                        boxes: Tensor, 
                        scores: Tensor, 
                        labels: Tensor,
                        boxes_actual: Tensor, 
                        labels_actual: Tensor,
                        figsize=(12, 6),
                        show=True):
    # Ensure img is a PIL Image
    if isinstance(img, str):
        img = Image.open(img)
    
    # Create a copy of the original image for predicted boxes
    img_pred = deepcopy(img)
    if img_pred.mode == 'L':
        img_pred = img_pred.convert("RGB")
    draw_pred = ImageDraw.Draw(img_pred)
    
    # Draw predicted boxes, scores, and labels
    for bbox, score, label in zip(boxes, scores, labels):
        x0, y0, w, h = bbox.tolist()
        draw_pred.rectangle((x0, y0, x0 + w, y0 + h), outline="red", width=3)
    
    # Create another copy of the original image for actual boxes
    img_actual = deepcopy(img)
    if img_actual.mode == 'L':
        img_actual = img_actual.convert("RGB")
    draw_actual = ImageDraw.Draw(img_actual)
    
    # Draw actual boxes and labels
    for bbox, label in zip(boxes_actual, labels_actual):
        x0, y0, w, h = bbox.tolist()
        draw_actual.rectangle((x0, y0, x0 + w, y0 + h), outline="green", width=3)
    
    # Display the images side by side
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(img_pred)
    ax[0].set_title('Predicted Boxes')
    ax[0].axis('off')
    
    ax[1].imshow(img_actual)
    ax[1].set_title('Actual Boxes')
    ax[1].axis('off')

    if show:
        plt.show()
    else:
        return fig


# if __name__ == "__main__":
#     home = os.environ["HOME"]
#     root = f"{home}/Datasets/flir/images_thermal_train"
#     with open(os.path.join(root, "coco.json"), "r") as oj:
#         coco = json.load(oj)
#     
#     view_boxes_from_coco_image_id(coco, 11, root)
    
