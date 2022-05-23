import torch
from torch import nn


class BaseCNN(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super(BaseCNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=3, padding='same'),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=3),
                                    nn.Conv2d(64, 128, kernel_size=3, padding='same'),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=3),
                                    nn.Conv2d(128, 256, kernel_size=3, padding='same'),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True)
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding='same'),
                                    nn.BatchNorm2d(256)
                                    )
        self.block3 = nn.Sequential(nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, kernel_size=3, padding='same'),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True)
                                    )
        self.block4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding='same'),
                                    nn.BatchNorm2d(512)
                                    )
        self.block5 = nn.Sequential(nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 1024, kernel_size=3, padding='same'),
                                    nn.BatchNorm2d(1024),
                                    nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
                                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x) + x
        x = self.block3(x)
        x = self.block4(x) + x
        x = self.block5(x)
        x = x.squeeze()  # 240, 1024
        return x