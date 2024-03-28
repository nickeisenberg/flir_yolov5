from .layers import Conv, SPPF, C3, UpAndCat
import torch
import torch.nn as nn


class YOLOv5(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.bb = nn.ModuleList([
            Conv(in_channels, 64, 6, 2, 2), 
            Conv(64, 128, 3, 2, 1),
            C3(128, 128, 3),
            Conv(128, 256, 3, 2, 1),
            C3(256, 256, 6),
            Conv(256, 512, 3, 2, 1),
            C3(512, 512, 9),
            Conv(512, 1024, 3, 2, 1),
            C3(1024, 1024, 3),
            SPPF(1024, 1024)
        ])
        self.bb_skips = []
        
        self.nc = UpAndCat()
        self.neck = nn.ModuleList([
            Conv(1024, 512, 1, 1, 0),
            C3(1024, 512, 3, False),
            Conv(512, 256, 1, 1, 0),
            C3(512, 256, 3, False),
        ])
        self.neck_skips = []
        
        self.predictions = []
        self.head = nn.ModuleList([
            nn.Conv2d(256, (5 + self.num_classes) * 3, 1, 1, 0),
            Conv(256, 256, 3, 2, 1),
            C3(512, 512, 3, False),
            nn.Conv2d(512, (5 + self.num_classes) * 3, 1, 1, 0),
            Conv(512, 512, 3, 2, 1),
            C3(1024, 1024, 3, False),
            nn.Conv2d(1024, (5 + self.num_classes) * 3, 1, 1, 0)
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.bb:
            x = layer(x)

            if isinstance(layer, C3) and x.shape[1] in [256, 512]:
                self.bb_skips.append(x)

        for layer in self.neck:
            x = layer(x)

            if isinstance(layer, Conv):
                self.neck_skips.append(x)
                x = self.nc(x, self.bb_skips.pop())

        for layer in self.head:
            if isinstance(layer, nn.Conv2d):
                pred = layer(x)
                self.predictions.insert(
                    0,
                    pred.view(
                        pred.shape[0], 3, 5 + self.num_classes, pred.shape[-2], pred.shape[-1]
                    ).permute(0, 1, 3, 4, 2)
                )
                continue

            x = layer(x)
            if isinstance(layer, Conv):
                x = torch.concat((x, self.neck_skips.pop()), 1)

        return tuple(self.predictions)



