import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3LossOriginal(nn.Module):
    """ ChatGPTs C to python translation of the YOLOv3 loss from the official
    implementation in the DarkNet framework
    """
    def __init__(self, num_classes, ignore_thresh=0.5, truth_thresh=1.0, 
                 object_scale=5.0, noobject_scale=1.0, class_scale=1.0, 
                 coord_scale=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

    def forward(self, pred, target, anchors, num_anchors, grid_size):
        batch_size = pred.size(0)
        stride = grid_size // pred.size(2)
        bbox_attrs = 5 + self.num_classes
        anchor_step = len(anchors) // num_anchors
        
        pred = pred.view(
            batch_size, num_anchors, bbox_attrs, grid_size, grid_size
        ).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]

        conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])

        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in anchors]

        obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(
            pred, target, scaled_anchors, num_anchors, grid_size
        )

        loss_x = self.coord_scale * F.mse_loss(x[obj_mask], tx[obj_mask], reduction='sum')
        loss_y = self.coord_scale * F.mse_loss(y[obj_mask], ty[obj_mask], reduction='sum')
        loss_w = self.coord_scale * F.mse_loss(w[obj_mask], tw[obj_mask], reduction='sum')
        loss_h = self.coord_scale * F.mse_loss(h[obj_mask], th[obj_mask], reduction='sum')

        loss_conf_obj = self.object_scale * F.binary_cross_entropy(
            conf[obj_mask], tconf[obj_mask], reduction='sum'
        )

        loss_conf_noobj = self.noobject_scale * F.binary_cross_entropy(
            conf[noobj_mask], tconf[noobj_mask], reduction='sum'
        )

        loss_cls = self.class_scale * F.binary_cross_entropy(
            pred_cls[obj_mask], tcls[obj_mask], reduction='sum'
        )

        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf_obj + loss_conf_noobj + loss_cls

        return total_loss

    def build_targets(self, pred, target, anchors, num_anchors, grid_size):
        batch_size = pred.size(0)
        anchor_step = len(anchors) // num_anchors

        obj_mask = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, dtype=torch.bool
        )

        noobj_mask = torch.ones(
            batch_size, num_anchors, grid_size, grid_size, dtype=torch.bool
        )
        tx = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, dtype=torch.float32
        )
        ty = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, dtype=torch.float32
        )
        tw = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, dtype=torch.float32
        )
        th = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, dtype=torch.float32
        )
        tconf = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, dtype=torch.float32
        )
        tcls = torch.zeros(
            batch_size, num_anchors, grid_size, grid_size, self.num_classes, 
            dtype=torch.float32
        )


        for b in range(batch_size):
            for t in range(target.size(1)):
                if target[b, t].sum() == 0:
                    continue
                gx = target[b, t, 1] * grid_size
                gy = target[b, t, 2] * grid_size
                gw = target[b, t, 3] * grid_size
                gh = target[b, t, 4] * grid_size

                gi = int(gx)
                gj = int(gy)
                best_n = -1
                best_iou = 0.0
                for n in range(num_anchors):
                    aw = anchors[n][0]
                    ah = anchors[n][1]
                    iou = self.bbox_iou([0, 0, gw, gh], [0, 0, aw, ah])
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n

                if best_iou > self.ignore_thresh:
                    noobj_mask[b, best_n, gj, gi] = 0

                if best_iou > self.truth_thresh:
                    obj_mask[b, best_n, gj, gi] = 1
                    noobj_mask[b, best_n, gj, gi] = 0
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][0] + 1e-16)
                    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][1] + 1e-16)
                    tconf[b, best_n, gj, gi] = 1
                    tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls

    def bbox_iou(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = self.xywh_to_xyxy(*box1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self.xywh_to_xyxy(*box2)
    
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
        return iou
    
    def xywh_to_xyxy(self, x, y, w, h):
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2


class YOLOv3Loss(nn.Module):
    """ ChatGPTs C to python translation of the YOLOv3 loss from the official
    implementation in the DarkNet framework with the addition of the iou used
    in the object loss confidence score
    """
    def __init__(self, num_classes, ignore_thresh=0.5, truth_thresh=1.0, 
                 object_scale=5.0, noobject_scale=1.0, class_scale=1.0, 
                 coord_scale=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

    def forward(self, pred, target, anchors, num_anchors, grid_size):
        batch_size = pred.size(0)
        stride = grid_size // pred.size(2)
        bbox_attrs = 5 + self.num_classes
        anchor_step = len(anchors) // num_anchors
        
        pred = pred.view(batch_size, num_anchors, bbox_attrs, grid_size, 
                         grid_size).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])

        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in anchors]

        obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(
            pred, target, scaled_anchors, num_anchors, grid_size)

        # Number of positive samples
        num_pos = torch.sum(obj_mask.float())

        # Calculate IoUs for objectness loss
        box_preds = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), 
                               h.unsqueeze(-1)), -1)
        ious = self.bbox_iou(box_preds[obj_mask], 
                             target[..., 1: 5][obj_mask]).detach()

        # Calculate losses and normalize by the number of positive samples
        loss_x = self.coord_scale * F.mse_loss(x[obj_mask], tx[obj_mask], 
                                               reduction='sum') / num_pos
        loss_y = self.coord_scale * F.mse_loss(y[obj_mask], ty[obj_mask], 
                                               reduction='sum') / num_pos
        loss_w = self.coord_scale * F.mse_loss(w[obj_mask], tw[obj_mask], 
                                               reduction='sum') / num_pos
        loss_h = self.coord_scale * F.mse_loss(h[obj_mask], th[obj_mask], 
                                               reduction='sum') / num_pos

        loss_conf_obj = self.object_scale * F.mse_loss(conf[obj_mask], 
                            ious * tconf[obj_mask], reduction='sum') / num_pos
        loss_conf_noobj = self.noobject_scale * F.binary_cross_entropy(
                            conf[noobj_mask], tconf[noobj_mask], 
                            reduction='sum') / (grid_size * grid_size * 
                            num_anchors - num_pos)
        
        loss_cls = self.class_scale * F.binary_cross_entropy(pred_cls[obj_mask], 
                                                             tcls[obj_mask], 
                                                             reduction='sum') / num_pos

        total_loss = (loss_x + loss_y + loss_w + loss_h + loss_conf_obj + 
                      loss_conf_noobj + loss_cls)

        return total_loss

    def build_targets(self, pred, target, anchors, num_anchors, grid_size):
        batch_size = pred.size(0)
        anchor_step = len(anchors) // num_anchors
        obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 
                               dtype=torch.bool)
        noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, 
                                dtype=torch.bool)
        tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 
                         dtype=torch.float32)
        ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 
                         dtype=torch.float32)
        tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 
                         dtype=torch.float32)
        th = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 
                         dtype=torch.float32)
        tconf = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 
                            dtype=torch.float32)
        tcls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 
                           self.num_classes, dtype=torch.float32)

        for b in range(batch_size):
            for t in range(target.size(1)):
                if target[b, t].sum() == 0:
                    continue
                gx = target[b, t, 1] * grid_size
                gy = target[b, t, 2] * grid_size
                gw = target[b, t, 3] * grid_size
                gh = target[b, t, 4] * grid_size

                gi = int(gx)
                gj = int(gy)
                best_n = -1
                best_iou = 0.0
                for n in range(num_anchors):
                    aw = anchors[n][0]
                    ah = anchors[n][1]
                    iou = self.bbox_iou([0, 0, gw, gh], [0, 0, aw, ah])
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n

                if best_iou > self.ignore_thresh:
                    noobj_mask[b, best_n, gj, gi] = 0

                if best_iou > self.truth_thresh:
                    obj_mask[b, best_n, gj, gi] = 1
                    noobj_mask[b, best_n, gj, gi] = 0
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][0] 
                                                      + 1e-16)
                    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][1] 
                                                      + 1e-16)
                    tconf[b, best_n, gj, gi] = 1
                    tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls

    def bbox_iou(self, box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = self.xywh_to_xyxy(*box1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self.xywh_to_xyxy(*box2)

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def xywh_to_xyxy(self, x, y, w, h):
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2
