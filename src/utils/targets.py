import torch
from torch.nn import Sigmoid

from ..utils import iou


def build_yolo_target(return_shape: tuple[tuple, ...],
                      bboxes: list[torch.Tensor], 
                      label_ids: list[int], 
                      normalized_anchors: torch.Tensor, 
                      scales: list[int],
                      img_width: int, 
                      img_height: int,
                      iou_thresh: float =0.5):

    assert len(return_shape) == len(scales)
    
    _target = []
    for shape in return_shape:
        _target.append(torch.zeros(shape))
    target: tuple[torch.Tensor, ...] = tuple(_target)

    anchors = normalized_anchors * torch.tensor([img_width, img_height])

    for bbox, label_id in zip(bboxes, label_ids):
        target = _populate_yolo_target_for_one_bbox(
            target=target, bbox=bbox, label_id=label_id, anchors=anchors,
            scales=scales, iou_thresh=iou_thresh
        )

    return target


def _populate_yolo_target_for_one_bbox(target: tuple[torch.Tensor, ...], 
                                       bbox: torch.Tensor, 
                                       label_id: int,
                                       anchors: torch.Tensor,
                                       scales: list[int],
                                       iou_thresh=0.5,
                                       by_center=False):

    x, y, w, h = bbox

    ious = torch.tensor([
        iou(anchor, bbox, share_center=True) 
        for anchor in anchors 
    ])
    ranked_iou_idxs = ious.argsort(descending=True)

    scale_is_assigned = [False, False, False]

    for idx in ranked_iou_idxs:

        scale_id = int(idx // len(scales))
        scale = scales[scale_id]

        anchor_id = int(idx % len(scales))

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

            width_cell, height_cell = w / scale, h / scale

            box_coordinates = torch.tensor(
                [x_cell, y_cell, width_cell, height_cell]
            )
            target[scale_id][anchor_id, row_id, col_id, 1:5] = box_coordinates
            target[scale_id][anchor_id, row_id, col_id, 5] = int(label_id)
            scale_is_assigned[scale_id] = True

        elif not is_taken and ious[idx] > iou_thresh:
            target[scale_id][anchor_id, row_id, col_id, 0] = -1

    return target


def decode_yolo_output(yolo_output: tuple[torch.Tensor, ...],
                       p_thresh: float,
                       normalized_anchors,
                       scales,
                       is_pred: bool = True):

    sigmoid = Sigmoid()

    boxes: list[float] = []
    labels: list[int] = []
    scores: list[float] = []
    locs: list[int] = []

    decoded_output = {
        "boxes": boxes, "labels": labels, "scores": scores, "locs": locs
    }

    for scale_id, t in enumerate(yolo_output):

        scale = scales[scale_id]
        scaled_ancs = normalized_anchors * torch.tensor(
            [img_width / scale, img_height / scale]
        )

        dims: list[tuple[torch.Tensor, ...]] = list(
            zip(*torch.where(t[..., 0:1] >= p_thresh)[:-1])
        )

        for dim in dims:
            if is_pred:
                batch_id, anc_id, row, col = dim

                bbox_info = t[dim][: 5]
                bbox_info[:3] = sigmoid(bbox_info[:3])

                p, x, y, w, h = bbox_info

                label_id = torch.argmax(t[dim][5:])

                x = (x + col.item()) * scale
                y = (y + row.item()) * scale

                w = torch.exp(w) * scaled_ancs[anc_id][0] * scale
                h = torch.exp(h) * scaled_ancs[anc_id][1] * scale

            else:
                anc_id, row, col = dim

                p, x, y, w, h, label_id = t[dim]
                x, y = (x + col.item()) * scale, (y + row.item()) * scale
                w = w * scale
                h = h * scale

            decoded_output['boxes'].append(
                [x.item(), y.item(), w.item(), h.item()]
            )
            decoded_output['labels'].append(int(label_id.item()))
            decoded_output['scores'].append(p.item())
            decoded_output['locs'].append(list(dim))

    for k in decoded_output.keys():
        decoded_output[k] = torch.tensor(decoded_output[k])

    ranked_inds = decoded_output['scores'].argsort(descending=True)

    for k in decoded_output.keys():
        decoded_output[k] = decoded_output[k][ranked_inds]

    return decoded_output


if __name__ == "__main__":
    pass

import os
from sys import path
path.append(f"{os.environ['HOME']}GitRepos/flir_yolov5")
import json
from src.utils.utils import make_yolo_anchors, scale_anchors, iou

home = os.environ["HOME"]
with open(f"{home}/Datasets/flir/images_thermal_train/coco.json", "r") as oj:
    coco = json.load(oj)

anchors = make_yolo_anchors(coco, 640, 512, 9)

return_shape = (
    (3, 16, 20, 6),
    (3, 32, 40, 6),
    (3, 64, 80, 6),
)
bboxes = [
    torch.tensor([100, 100, 50 ,50]),
    torch.tensor([200, 200, 40 ,40])
]
label_ids = [1, 2]
scales = [32, 16, 8]
img_width = 640
img_height = 512


target = build_yolo_target(
    return_shape, bboxes, label_ids, anchors, scales, img_width, img_height
)

batched_target = tuple([t.unsqueeze(0) for t in target])

for t in target:
    dims: list[tuple[torch.Tensor, ...]] = list(
        zip(*torch.where(t[..., 0:1] >= .8)[:-1])
    )
    for dim in dims:
        print(t[dim])
        break
    break

decoded = decode_yolo_output(target, .8, anchors, scales, False)

decoded = decode_yolo_output(batched_target, .8, anchors, scales, True)

decoded["boxes"]




