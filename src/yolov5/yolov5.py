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
            SPPF(1024, 1024),
        ])
        
        self.nc = UpAndCat()
        self.neck = nn.ModuleList([
            Conv(1024, 512, 1, 1, 0),
            C3(1024, 512, 3, False),
            Conv(512, 256, 1, 1, 0),
            C3(512, 256, 3, False),
        ])
        
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
        device = x.device.type

        bb_skips = []
        neck_skips = []

        predictions = []
        for layer in self.bb:
            x = layer(x)

            if isinstance(layer, C3) and x.shape[1] in [256, 512]:
                bb_skips.append(x)
        
        for layer in self.neck:
            x = layer(x)
            if isinstance(layer, Conv):
                neck_skips.append(x)
                x = self.nc(x, bb_skips.pop().to(device))

        for layer in self.head:
            if isinstance(layer, nn.Conv2d):
                pred = layer(x)
                predictions.insert(
                    0,
                    pred.view(
                        pred.shape[0], 
                        3, 
                        5 + self.num_classes, 
                        pred.shape[-2], 
                        pred.shape[-1]
                    ).permute(0, 1, 3, 4, 2)
                )
                continue

            x = layer(x)
            if isinstance(layer, Conv):
                x = torch.concat((x, neck_skips.pop().to(device)), 1)

        return tuple(predictions)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, use_bn=True, act=nn.SiLU(), **kwargs):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, bias=False, **kwargs)
        self.act = act if act is not None else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.use_bn:
            return self.act(self.bn(self.conv(x)))  
        else: 
            return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        _c = out_channels // 2
        self.conv1 = Conv(in_channels, _c, 1, 1, 0)
        self.conv2 = Conv(_c, out_channels, 3, 1, 1)
        self.residual = residual and in_channels == out_channels

    def forward(self, input):
        x = self.conv2(self.conv1(input))
        return input + x if self.residual else x


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        _c = in_channels // 2
        self.conv1 = Conv(in_channels, _c, 1, 1, 0)
        self.conv2 = Conv(_c * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size, 1, kernel_size // 2)

    def forward(self, input):
        x = [self.conv1(input)]
        x.extend(self.m(x[-1]) for _ in range(3))
        return self.conv2(torch.cat(x, 1))


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats, residual=True):
        super().__init__()
        _c = out_channels // 2
        self.conv1 = Conv(in_channels, _c, 1, 1, 0)
        self.conv2 = Conv(in_channels, _c, 1, 1, 0)
        self.conv3 = Conv(2 * _c, out_channels, 1, 1, 0)
        self.b = nn.Sequential(
            *[Bottleneck(_c, _c, residual) for _ in range(num_repeats)]
        )

    def forward(self, x):
        return self.conv3(torch.cat((self.b(self.conv1(x)), self.conv2(x)), 1))


class UpAndCat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, skip):
        x = nn.Upsample(tuple(skip.shape[-2:]))(x)
        return torch.cat((x, skip), 1)


if __name__ == "__main__":
    yolo_v5 = YOLOv5(3, 5)
    yolo_v5 = yolo_v5.eval()
    with torch.no_grad():
        output = yolo_v5(torch.randn((1, 3, 512, 640)))
    for scale in output:
        print(scale.shape)
