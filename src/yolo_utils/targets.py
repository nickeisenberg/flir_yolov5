from typing import cast
import torch
from torch import Tensor
from torch.nn import Sigmoid
from copy import deepcopy

from ..yolo_utils.utils import iou, nms

def build_yolo_target(return_shape: tuple[tuple, ...],
                      bboxes: list[torch.Tensor] | torch.Tensor, 
                      label_ids: list[int] | torch.Tensor, 
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
            target=target, bbox=bbox, label_id=int(label_id), anchors=anchors,
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


def decode_yolo_tuple(yolo_tuple: tuple[Tensor, ...],
                      img_width: int,
                      img_height: int,
                      normalized_anchors: Tensor,
                      scales: list[int],
                      score_thresh: float | None = None,
                      nms_iou_thresh: float | None = None,
                      min_box_dim: tuple[int, int] | None = None,
                      is_pred: bool = True) -> list[dict[str, Tensor]]:
    """ Decode a yolo prediction tuple or a yolo target into a dictionary
    with keys boxes, labels and scores. The scores key will be ignored in the
    case that the yolo tuple is a target
    """

    sigmoid = Sigmoid()

    batch_size = yolo_tuple[0].size(0)
    for scale_pred in yolo_tuple:
        assert scale_pred.size(0) == batch_size

    _boxes: list[list[float]] = []
    _labels: list[int] = [] 
    _scores: list[float] = [] 

    decoded_all_images = []
    for _ in range(batch_size):
        if is_pred:
            decoded_image = {
                "boxes": deepcopy(_boxes), 
                "labels": deepcopy(_labels), 
                "scores": deepcopy(_scores), 
            }
        else:
            decoded_image = {
                "boxes": deepcopy(_boxes), 
                "labels": deepcopy(_labels), 
            }
        decoded_all_images.append(decoded_image)

    for scale_id, t in enumerate(yolo_tuple):

        scale = scales[scale_id]
        scaled_ancs = normalized_anchors[3 * scale_id: 3 * (scale_id + 1)] * torch.tensor(
            [img_width / scale, img_height / scale]
        )
        
        if is_pred:
            dims_where: list[tuple[torch.Tensor, ...]] = list(
                zip(*torch.where(t[..., 0:1] >= score_thresh)[:-1])
            )
        else:
            dims_where: list[tuple[torch.Tensor, ...]] = list(
                zip(*torch.where(t[..., 0:1] >= .8)[:-1])
            )

        for dim in dims_where:
            if is_pred:
                batch_id, anc_id, row, col = dim

                bbox_info = deepcopy(t[dim][: 5].detach())

                bbox_info[:3] = sigmoid(bbox_info[:3])

                p, x, y, w, h = bbox_info

                label_id = torch.argmax(t[dim][5:])

                x = (x + col.item()) * scale
                y = (y + row.item()) * scale

                w = torch.exp(w) * scaled_ancs[anc_id][0] * scale
                h = torch.exp(h) * scaled_ancs[anc_id][1] * scale

                bbox = [x.item(), y.item(), w.item(), h.item()]
                label = int(label_id.item())
                score = p.item()

                if min_box_dim is not None:
                    if bbox[2] < min_box_dim[0] or bbox[3] < min_box_dim[1]:
                        continue

                decoded_all_images[batch_id]["boxes"].append(bbox)
                decoded_all_images[batch_id]["labels"].append(label)
                decoded_all_images[batch_id]["scores"].append(score)

            else:
                batch_id, anc_id, row, col = dim

                p, x, y, w, h, label_id = deepcopy(t[dim].detach())

                x, y = (x + col.item()) * scale, (y + row.item()) * scale
                w = w * scale
                h = h * scale

                bbox = [x.item(), y.item(), w.item(), h.item()]
                label = int(label_id.item())
                
                if not bbox in decoded_all_images[batch_id]['boxes']:
                    decoded_all_images[batch_id]["boxes"].append(bbox)
                    decoded_all_images[batch_id]["labels"].append(label)
    
    for decoded_image in decoded_all_images:
        for key in decoded_image.keys():
            decoded_image[key] = torch.tensor(decoded_image[key])

    if is_pred:
        for decoded_image in decoded_all_images:
            ranked_inds = decoded_image['scores'].argsort(descending=True)
            decoded_image["boxes"] = decoded_image["boxes"][ranked_inds]

            nms_inds = nms(decoded_image["boxes"], nms_iou_thresh)
            decoded_image["boxes"] = decoded_image["boxes"][nms_inds]

            for k in decoded_image.keys():
                if k == "boxes":
                    continue
                decoded_image[k] = decoded_image[k][ranked_inds]
                decoded_image[k] = decoded_image[k][nms_inds]

    return decoded_all_images
