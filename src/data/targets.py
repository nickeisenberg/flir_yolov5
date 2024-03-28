from typing import DefaultDict, Iterable
import torch
from torch.nn import Sigmoid, Module
from copy import deepcopy

from ..utils import iou, scale_anchors


def build_yolo_target(return_shape: tuple[tuple, ...],
                      bboxes: list[torch.Tensor], 
                      label_ids: list[int], 
                      normalized_anchors: list[torch.Tensor], 
                      scales: list[int],
                      img_width: int, 
                      img_height: int,
                      iou_thresh: float =0.5):

    assert len(return_shape) == len(scales)
    
    _target = []
    for shape in return_shape:
        _target.append(torch.zeros(shape))
    target: tuple[torch.Tensor, ...] = tuple(_target)

    anchors = scale_anchors(
        normalized_anchors, 1, img_width, img_height
    )

    for bbox, label_id in zip(bboxes, label_ids):
        target = _populate_yolo_target_for_one_bbox(
            target=target, bbox=bbox, label_id=label_id, anchors=anchors,
            scales=scales, iou_thresh=iou_thresh
        )


def _populate_yolo_target_for_one_bbox(target: tuple[torch.Tensor, ...], 
                                       bbox: torch.Tensor, 
                                       label_id: int,
                                       anchors: list[torch.Tensor],
                                       scales: list[int],
                                       iou_thresh=0.5,
                                       by_center=False):

    x, y, w, h = bbox

    ious = torch.tensor([
        iou(anchor, bbox, share_center=True) 
        for anchor in anchors 
    ])
    ranked_inds = ious.argsort(descending=True)

    scale_is_assigned = [False, False, False]

    for ranked_ind in ranked_inds:

        scale_id = int(ranked_ind // len(scales))
        scale = scales[scale_id]

        anchor_id = int(ranked_ind % len(scales))

        if by_center:
            row_id = int((y + (h // 2)) // scale)
            col_id = int((x + (w // 2)) // scale)
        else:
            row_id = int(y // scale)
            col_id = int(x // scale)

        is_taken = target[scale_id][anchor_id, row_id, col_id, 0]

        if not is_taken and not scale_is_assigned[scale_id]:
            target[scale_id][anchor_id, row_id, col_id, 0] = 1
            x_cell, y_cell = x / scale - col_id, y / scale - row_id
            width_cell, height_cell = (
                w / scale,
                h / scale,
            )
            box_coordinates = torch.tensor(
                [x_cell, y_cell, width_cell, height_cell]
            )
            target[scale_id][anchor_id, row_id, col_id, 1:5] = box_coordinates
            target[scale_id][anchor_id, row_id, col_id, 5] = int(label_id)
            scale_is_assigned[scale_id] = True

        elif not is_taken and ious[ranked_ind] > iou_thresh:
            target[scale_id][anchor_id, row_id, col_id, 0] = -1

    return target







def decode_yolo_target():
    pass


class YOLOTarget:
    def __init__(self, 
                 global_class_id_to_local_id: dict, 
                 anchors: torch.Tensor, 
                 scales: list, 
                 img_w: int, 
                 img_h: int):

        self.global_class_id_to_local_id = global_class_id_to_local_id 
        self.global_class_id_to_local_id_inv = {
            v: k for k, v in global_class_id_to_local_id.items()
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

        local_id = self.global_class_id_to_local_id[cat_id]

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

            is_taken = target[scale_id][anchor_id, row_id, col_id, 0]

            if not is_taken and not scale_is_assigned[scale_id]:
                target[scale_id][anchor_id, row_id, col_id, 0] = 1
                x_cell, y_cell = x / scale - col_id, y / scale - row_id
                width_cell, height_cell = (
                    w / scale,
                    h / scale,
                )
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                target[scale_id][anchor_id, row_id, col_id, 1:5] = box_coordinates
                target[scale_id][anchor_id, row_id, col_id, 5] = int(local_id)
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
                    cat = self.global_class_id_to_local_id_inv[cat.item()]
                else:
                    x, y, w, h, p, cat = t[dim]
                    x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
                    w = w * scale
                    h = h * scale
                    cat = self.global_class_id_to_local_id_inv[cat.item()]
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

