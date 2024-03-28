import torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import torch


def scale_anchors(anchors, scale, img_w, img_h, device="cpu"):
    scaler = torch.tensor([
        img_w / min(scale, img_w), 
        img_h / min(scale, img_h)
    ]).to(device)
    scaled_anchors = anchors * scaler
    return scaled_anchors


def iou(box1, box2, share_center=False):
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
        intersection_a = min(box1[..., -2], box2[..., -2]) * min(box1[..., -1], box2[..., -1])
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


class ConstructAnchors:
    def __init__(self, coco, img_width, img_height, n_clusters=9):
        if isinstance(coco, str): 
            with open(coco, 'r') as oaf:
                self.coco = json.load(oaf)
        elif isinstance(coco, dict):
            self.coco = coco

        self.bboxes = np.array([
            [x['bbox'][2] / img_width, x['bbox'][3] / img_height]  
            for x in self.coco['annotations']
        ])
        self._k_means(n_clusters)

    def _k_means(self, n_clusters=9):
        self._KMeans = KMeans(n_clusters=n_clusters)
        self._clusters = self._KMeans.fit_predict(self.bboxes)
        cluster_centers = self._KMeans.cluster_centers_
        # sorted_args = np.argsort(np.linalg.norm(cluster_centers, axis=1))[::-1]
        sorted_args = np.argsort(
            cluster_centers[:,0] * cluster_centers[:, 1]
        )[::-1]
        self.anchors = torch.tensor(np.hstack(
            (sorted_args.reshape((-1, 1)), cluster_centers[sorted_args])
        ))
        return None

    def view_clusters(self, show=True):
        fig = plt.figure()
        plt.scatter(self.bboxes[:, 0], self.bboxes[:, 1], c=self._clusters)
        if show:
            plt.show()
        else:
            return fig
