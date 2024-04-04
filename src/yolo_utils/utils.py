import torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import torch


def scale_anchors(anchors: torch.Tensor, 
                  scale: int, 
                  img_w: int, 
                  img_h: int, 
                  device="cpu"):

    scaler = torch.tensor([
        img_w / min(scale, img_w), 
        img_h / min(scale, img_h)
    ]).to(device)
    scaled_anchors = anchors * scaler
    return scaled_anchors


def iou(box1: torch.Tensor, box2:torch.Tensor, share_center=False):
    """
    Parameters
    ----------
    box1: torch.Tensor
        Iterable of format [bx, by, bw, bh] where bx and by are the coords of
        the top left of the bounding box and bw and bh are the width and
        height
    box2: same as box1
    pred: boolean default = False
        If False, then the assumption is made that the boxes share the same
        center.
    """
    ep = 1e-6

    if share_center:
        box1_a = box1[..., -2] * box1[..., -1]
        box2_a = box2[..., -2] * box2[..., -1]

        intersection_a = torch.min(box1[..., -2], box2[..., -2]).item()
        intersection_a *= torch.min(box1[..., -1], box2[..., -1]).item()

        union_a = box1_a + box2_a - intersection_a
        return intersection_a / union_a
    
    else:
        len_x = torch.sub(
            torch.min(
                box1[..., 0: 1] + box1[..., 2: 3], 
                box2[..., 0: 1] + box2[..., 2: 3]
            ),
            torch.max(box1[..., 0: 1], box2[..., 0: 1])
        ).clamp(0)

        len_y = torch.sub(
            torch.min(
                box1[..., 1: 2] + box1[..., 3: 4], 
                box2[..., 1: 2] + box2[..., 3: 4]
            ),
            torch.max(box1[..., 1: 2], box2[..., 1: 2])
        ).clamp(0)

        box1_a = box1[..., 2: 3] * box1[..., 3: 4]
        box2_a = box2[..., 2: 3] * box2[..., 3: 4]

        intersection_a = len_x * len_y

        union_a = box1_a + box2_a - intersection_a + ep

        return intersection_a / union_a


def make_yolo_anchors(coco: str | dict, 
                      img_width: int, 
                      img_height: int, 
                      n_clusters=9,
                      view_clusters=False):
    if isinstance(coco, str):
        try:
            with open(coco, "r") as oj:
                coco = json.load(oj)
        except Exception as e:
            raise e

    assert type(coco) == dict

    scaled_bbox_dims = np.array([
        [x['bbox'][2] / img_width, x['bbox'][3] / img_height]  
        for x in coco['annotations']
    ])


    k_means = KMeans(n_clusters=n_clusters)
    _ = k_means.fit(scaled_bbox_dims)
    cluster_centers = k_means.cluster_centers_

    sorted_args = np.argsort(
        cluster_centers[:,0] * cluster_centers[:, 1]
    )[::-1]

    anchors = torch.tensor(cluster_centers[sorted_args])
    
    if not view_clusters:
        return anchors

    else:
        clusters = k_means.predict(scaled_bbox_dims)
        fig = plt.figure()
        plt.scatter(scaled_bbox_dims[:, 0], scaled_bbox_dims[:, 1], c=clusters)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="o")
        plt.show()
        return anchors


def nms(boxes: torch.Tensor, threshold):
    """
    Apply non-maximum suppression to suppress overlapping bounding boxes
    :param boxes: List of bounding boxes in the format [x, y, w, h]
    :param scores: List of scores for each bounding box
    :param threshold: IoU threshold for overlapping boxes
    :return: Indices of boxes to keep
    """

    box_list = boxes.tolist()

    idxs = [*range(len(boxes))]
    keep = []

    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [idx for idx in idxs if iou(
            torch.tensor(box_list[current]), torch.tensor(box_list[idx])
        ) < threshold]

    return keep


if __name__ == "__main__":
    pass

    import os
    
    home = os.environ["HOME"]
    with open(f"{home}/Datasets/flir/images_thermal_train/coco.json", "r") as oj:
        coco = json.load(oj)
    
    
    anchors = make_yolo_anchors(coco, 640, 512, 9, True)
