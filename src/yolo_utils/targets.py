import torch
from torch import Tensor
from torch.nn import Sigmoid

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
                      iou_thresh: float | None = None,
                      is_pred: bool = True) -> list[dict[str, Tensor]]:
    """ Decode a yolo prediction tuple or a yolo target into a dictionary
    with keys boxes, labels and scores. The scores key will be ignored in the
    case that the yolo tuple is a target
    """

    sigmoid = Sigmoid()

    _boxes: list[list[float]] = []
    _labels: list[int] = []
    _scores: list[float] = []
    
    if is_pred:
        decoded_image = {
            "boxes": _boxes, 
            "labels": _labels, 
            "scores": _scores, 
        }
    else:
        decoded_image = {
            "boxes": _boxes, 
            "labels": _labels, 
        }

    batch_size = yolo_tuple[0].size(0)
    for scale_pred in yolo_tuple:
        assert scale_pred.size(0) == batch_size

    decoded_all_images = [decoded_image for _ in range(batch_size)]

    for scale_id, t in enumerate(yolo_tuple):

        scale = scales[scale_id]
        scaled_ancs = normalized_anchors * torch.tensor(
            [img_width / scale, img_height / scale]
        )
        scaled_ancs = scaled_ancs[3 * scale_id: 3 * (scale_id + 1)]
        
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

                bbox_info = t[dim][: 5]
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

                decoded_all_images[batch_id]["boxes"].append(bbox)
                decoded_all_images[batch_id]["labels"].append(label)
                decoded_all_images[batch_id]["scores"].append(score)

            else:
                batch_id, anc_id, row, col = dim

                p, x, y, w, h, label_id = t[dim]
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
            nms_inds = nms(decoded_image["boxes"], iou_thresh)
            decoded_image["boxes"] = decoded_image["boxes"][nms_inds]

            for k in decoded_image.keys():
                if k == "boxes":
                    continue
                decoded_image[k] = decoded_image[k][ranked_inds]
                decoded_image[k] = decoded_image[k][nms_inds]

    return decoded_all_images


if __name__ == "__main__":
    import os
    from sys import path
    path.append(f"{os.environ['HOME']}GitRepos/flir_yolov5")
    import json
    from src.yolo_utils.utils import make_yolo_anchors, scale_anchors, iou
    
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
    
    decoded = decode_yolo_output(target, img_width, img_height, .8, anchors, scales, False)
    decoded["boxes"]
    
    decoded = decode_yolo_output(batched_target, img_width, img_height, .8, anchors, scales, True)
    decoded["boxes"]
