import torch
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision as _MeanAvergePrecision
from torchmetrics.detection.helpers import _input_validator


class MeanAveragePrecision(_MeanAvergePrecision):
    def __init__(self,
                 box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
                 iou_type: Union[Literal["bbox", "segm"], Tuple[str]] = "bbox",
                 iou_thresholds: Optional[List[float]] = None,
                 rec_thresholds: Optional[List[float]] = None,
                 max_detection_thresholds: Optional[List[int]] = None,
                 class_metrics: bool = False,
                 extended_summary: bool = False,
                 average: Literal["macro", "micro"] = "macro",
                 backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
                 **kwargs: Any):
        super().__init__(
            box_format=box_format,
            iou_type=iou_type,
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            class_metrics=class_metrics,
            extended_summary=extended_summary,
            average=average,
            backend=backend,
            **kwargs)

    def update(self, 
               preds: List[Dict[str, Tensor]], 
               target: List[Dict[str, Tensor]]) -> None:
            """Update metric state.
    
            Raises:
                ValueError:
                    If ``preds`` is not of type (:class:`~List[Dict[str, Tensor]]`)
                ValueError:
                    If ``target`` is not of type ``List[Dict[str, Tensor]]``
                ValueError:
                    If ``preds`` and ``target`` are not of the same length
                ValueError:
                    If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length
                ValueError:
                    If any of ``target.boxes`` and ``target.labels`` are not of the same length
                ValueError:
                    If any box is not type float and of length 4
                ValueError:
                    If any class is not type int and of length 1
                ValueError:
                    If any score is not type float and of length 1
    
            """
            _input_validator(preds, target, iou_type=self.iou_type)
    
            for item in preds:
                bbox_detection, mask_detection = self._get_safe_item_values(
                item, warn=self.warn_on_many_detections
            )
                if bbox_detection is not None:
                    self.detection_box.append(bbox_detection)
                if mask_detection is not None:
                    self.detection_mask.append(mask_detection)
                self.detection_labels.append(item["labels"])
                self.detection_scores.append(item["scores"])
    
            for item in target:
                bbox_groundtruth, mask_groundtruth = self._get_safe_item_values(item)
                if bbox_groundtruth is not None:
                    self.groundtruth_box.append(bbox_groundtruth)
                if mask_groundtruth is not None:
                    self.groundtruth_mask.append(mask_groundtruth)
                self.groundtruth_labels.append(item["labels"])
                self.groundtruth_crowds.append(
                item.get("iscrowd", torch.zeros_like(item["labels"]))
            )
                self.groundtruth_area.append(
                item.get("area", torch.zeros_like(item["labels"]))
            )







