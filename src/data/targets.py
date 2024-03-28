from typing import DefaultDict
import torch
from torch.nn import Sigmoid
from copy import deepcopy

from ..utils import iou, scale_anchors


class YOLOTarget:
    """
    A class to build the target for a single image. The annotations for a
    single image should be in the following form:

    img_annotes = {
        boxes: [
            [x, y, w, h],  # bbox_0
            [x, y, w, h],  # bbox_1
            ...
        ],
        labels: [
            bbox_label_0, 
            bbox_label_1, 
            ...
        ]
    }

    Anchors should be vertically stacked as a tensor in decreasing scale and 
    area. Ie:

    anchors = torch.tensor([
        [w00, h00],  # scale_0
        [w01, h01],  # scale_0
        ...,
        [wi0, hi0],  # scale_i
        ...
        [wNK, hNK],  # scale_N
    ])

    and we need that wik * hik > wjp * hjp whenever i < j and k < p.
    

    Parameters:
    ----------
    category_id_mapper: dict
        see class_id_mapper of trfc.dataset.COCODataset

    """
    
    def __init__(self, 
                 class_id_mapper: dict, 
                 anchors: torch.Tensor, 
                 scales: list, 
                 img_w: int, 
                 img_h: int):

        self.class_id_mapper = class_id_mapper
        self.class_id_mapper_inv = {
            v: k for k, v in class_id_mapper.items()
        }
        self.anchors = anchors
        self.full_scale_anchors = scale_anchors(anchors, 1, img_w, img_h)
        self.scales = scales 

        self._target = (
            torch.zeros((3, 16, 20, 6)),
            torch.zeros((3, 32, 40, 6)),
            torch.zeros((3, 64, 80, 6))
        )


    def build(self, annotes: dict, debug=False):
        """
        Loops through all annotations for an image and builds the target.
        The result will be added to self.target.

        """
        _target = deepcopy(self._target)

        debug_tracker = []

        for bbox, cat_id in zip(annotes['boxes'], annotes['labels']):
            if debug:
                scales_predicted = self._populate_target_for_one_bbox(
                    bbox, cat_id.item(), _target, debug=True
                )
                debug_tracker.append((bbox, scales_predicted))
            else:
               self._populate_target_for_one_bbox(bbox, cat_id.item(), _target)
        
        if debug:
            return (_target, debug_tracker)
        else:
            return _target


    def _populate_target_for_one_bbox(
        self, bbox, cat_id: int, 
        target: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        by_center=False, debug=False):

        cat_id = self.class_id_mapper[cat_id]

        x, y, w, h = bbox

        ious = torch.tensor([
            iou(anchor, bbox, share_center=True) 
            for anchor in self.full_scale_anchors
        ])
        ranked_inds = ious.argsort(descending=True)

        iou_thresh = .5

        scale_is_assigned = [False, False, False]

        for ranked_ind in ranked_inds:
            scale_id = int(ranked_ind // len(self.scales))
            scale = self.scales[scale_id]
            anchor_id = int(ranked_ind % len(self.scales))

            if by_center:
                row_id = int((y + (h // 2)) // scale)
                col_id = int((x + (w // 2)) // scale)
            else:
                row_id = int(y // scale)
                col_id = int(x // scale)

            is_taken = target[scale_id][anchor_id, row_id, col_id, 4]

            if not is_taken and not scale_is_assigned[scale_id]:
                target[scale_id][anchor_id, row_id, col_id, 4] = 1
                x_cell, y_cell = x / scale - col_id, y / scale - row_id
                width_cell, height_cell = (
                    w / scale,
                    h / scale,
                )
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                target[scale_id][anchor_id, row_id, col_id, 0:4] = box_coordinates
                target[scale_id][anchor_id, row_id, col_id, 5] = int(cat_id)
                scale_is_assigned[scale_id] = True

            elif not is_taken and ious[ranked_ind] > iou_thresh:
                target[scale_id][anchor_id, row_id, col_id, 0] = -1

        if debug:
            return scale_is_assigned


    def decode_tuple(self, tup, p_thresh, is_pred):
        if not is_pred:
            p_thresh = 1

        sigmoid = Sigmoid()

        decoded = {"boxes": [], "labels": [], "scores": [], "locs": []}
        for scale_id, t in enumerate(tup):
            scale = self.scales[scale_id]
            scaled_ancs = scale_anchors(
                self.anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
            )
            dims = list(zip(*torch.where(t[..., 4:5] >= p_thresh)[:-1]))
            for dim in dims:
                if is_pred:
                    bbox_info = t[dim][: 5]
                    bbox_info[:2] = sigmoid(bbox_info[:2])
                    bbox_info[4] = sigmoid(bbox_info[4])
                    x, y, w, h, p = bbox_info
                    cat = torch.argmax(t[dim][5:])
                    x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
                    w = torch.exp(w) * scaled_ancs[dim[0]][0] * scale
                    h = torch.exp(h) * scaled_ancs[dim[0]][1] * scale
                    cat = self.class_id_mapper_inv[cat.item()]
                else:
                    x, y, w, h, p, cat = t[dim]
                    x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
                    w = w * scale
                    h = h * scale
                    cat = self.class_id_mapper_inv[cat.item()]
                decoded['boxes'].append(
                    [x.item(), y.item(), w.item(), h.item()]
                )
                decoded['labels'].append(cat)
                decoded['scores'].append(p.item())
                decoded['locs'].append(dim)
        
        for k in decoded.keys():
            decoded[k] = torch.tensor(decoded[k])

        ranked_inds = decoded['scores'].argsort(descending=True)
    
        return decoded

