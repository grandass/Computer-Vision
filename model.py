import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --------------------------
# Encoder
# --------------------------
class EfficientNetEncoder(nn.Module):
    def __init__(self, in_ch=1, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = efficientnet_b0(weights=weights)
        if in_ch != 3:
            # replace first conv to accept in_ch while keeping feature sizing
            effnet.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = effnet.features

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in [1, 2, 3, 4, 6]:
                skips.append(x)
        # expected skips channels (torchvision effnet_b0): [16, 24, 40, 80, 192]
        return skips  # skip0, skip1, skip2, skip3, skip4

# --------------------------
# ConvBlock used in decoder
# --------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.2):
        super().__init__()
        # GroupNorm(group_count, num_channels) requires num_channels % group_count == 0.
        # We use 8 groups, which divides the channel numbers used below.
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)

# --------------------------
# Full FCN with proper channel math and upsampling
# --------------------------
class EfficientNetFCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dropout=0.2):
        super().__init__()
        self.encoder = EfficientNetEncoder(in_ch=in_ch, pretrained=True)

        # Based on encoder skip channels: skip0=16, skip1=24, skip2=40, skip3=80, skip4=192
        # Stage 1: combine upsampled skip4 (192) with skip3 (80) -> 272 in
        self.c4 = ConvBlock(192 + 80, 128, dropout=dropout)

        # Stage 2: upsample c4 (128) concat skip2 (40) -> 168 in
        self.c3 = ConvBlock(128 + 40, 64, dropout=dropout)

        # Stage 3: upsample c3 (64) concat skip1 (24) -> 88 in
        self.c2 = ConvBlock(64 + 24, 32, dropout=dropout)

        # Stage 4: upsample c2 (32) concat skip0 (16) -> 48 in
        self.c1 = ConvBlock(32 + 16, 32, dropout=dropout)

        # final conv -> produce logits at full input resolution later
        self.head = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        input_size = x.shape[2:]  # keep original input size for final upsample
        skips = self.encoder(x)
        skip0, skip1, skip2, skip3, skip4 = skips  # channels: 16,24,40,80,192

        # Stage 1: skip4 (8x8) -> match skip3 (16x16)
        y = F.interpolate(skip4, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, skip3], dim=1)  # 192 + 80 = 272 channels
        y = self.c4(y)                     # out: 128 channels @ 16x16

        # Stage 2: upsample to skip2 (32x32)
        y = F.interpolate(y, size=skip2.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, skip2], dim=1)  # 128 + 40 = 168 channels
        y = self.c3(y)                     # out: 64 channels @ 32x32

        # Stage 3: upsample to skip1 (64x64)
        y = F.interpolate(y, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, skip1], dim=1)  # 64 + 24 = 88 channels
        y = self.c2(y)                     # out: 32 channels @ 64x64

        # Stage 4: upsample to skip0 (128x128)
        y = F.interpolate(y, size=skip0.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, skip0], dim=1)  # 32 + 16 = 48 channels
        y = self.c1(y)                     # out: 32 channels @ 128x128

        # Final upsample to input size (e.g., 256x256) and head
        y = F.interpolate(y, size=input_size, mode='bilinear', align_corners=False)  # now same as input
        logits = self.head(y)  # [B, out_ch, H, W]
        return logits

# --------------------------
# Student wrapper
# --------------------------
class StudentModel(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dropout=0.2):
        super().__init__()
        self.net = EfficientNetFCN(in_ch=in_ch, out_ch=out_ch, dropout=dropout)

    def forward(self, x):
        return self.net(x)
