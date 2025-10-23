import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class FCNBaseline(nn.Module):
    """A minimal FCN with light encoder-decoder and skip connections."""
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        # Encoder
        self.c1 = ConvBlock(in_ch, base_ch)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(base_ch, base_ch*2)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(base_ch*2, base_ch*4)
        # Decoder
        self.u2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.c4 = ConvBlock(base_ch*4, base_ch*2)
        self.u1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.c5 = ConvBlock(base_ch*2, base_ch)
        self.out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(self.p1(x1))
        x3 = self.c3(self.p2(x2))
        y = self.u2(x3)
        y = torch.cat([y, x2], dim=1)
        y = self.c4(y)
        y = self.u1(y)
        y = torch.cat([y, x1], dim=1)
        y = self.c5(y)
        logits = self.out(y)  # [B,1,H,W]
        return logits

class StudentModel(nn.Module):
    """Students: replace the backbone with your own architecture."""
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.net = FCNBaseline(in_ch, base_ch)
    def forward(self, x):
        return self.net(x)
