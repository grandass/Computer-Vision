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
            effnet.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = effnet.features

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in [1, 2, 3, 4, 6]:
                skips.append(x)
        # skip0, skip1, skip2, skip3, skip4 channels: 16,24,40,80,192
        return skips

# --------------------------
# ConvBlock used in decoder
# --------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.2):
        super().__init__()
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
# Attention Gate
# --------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# --------------------------
# Full FCN with attention
# --------------------------
class EfficientNetFCN_Attention(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dropout=0.2):
        super().__init__()
        self.encoder = EfficientNetEncoder(in_ch=in_ch, pretrained=True)

        # Conv blocks
        self.c4 = ConvBlock(192 + 80, 128, dropout=dropout)
        self.c3 = ConvBlock(128 + 40, 64, dropout=dropout)
        self.c2 = ConvBlock(64 + 24, 32, dropout=dropout)
        self.c1 = ConvBlock(32 + 16, 32, dropout=dropout)

        # Attention gates (F_g = decoder channels, F_l = skip channels)
        self.att4 = AttentionGate(F_g=192, F_l=80, F_int=64)
        self.att3 = AttentionGate(F_g=128, F_l=40, F_int=32)
        self.att2 = AttentionGate(F_g=64, F_l=24, F_int=16)
        self.att1 = AttentionGate(F_g=32, F_l=16, F_int=16)

        # Final conv
        self.head = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        skip0, skip1, skip2, skip3, skip4 = self.encoder(x)

        # Stage 1: skip4 -> skip3
        y = F.interpolate(skip4, size=skip3.shape[2:], mode='bilinear', align_corners=False)
        skip3_att = self.att4(skip3, y)
        y = torch.cat([y, skip3_att], dim=1)
        y = self.c4(y)

        # Stage 2: skip3 -> skip2
        y = F.interpolate(y, size=skip2.shape[2:], mode='bilinear', align_corners=False)
        skip2_att = self.att3(skip2, y)
        y = torch.cat([y, skip2_att], dim=1)
        y = self.c3(y)

        # Stage 3: skip2 -> skip1
        y = F.interpolate(y, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        skip1_att = self.att2(skip1, y)
        y = torch.cat([y, skip1_att], dim=1)
        y = self.c2(y)

        # Stage 4: skip1 -> skip0
        y = F.interpolate(y, size=skip0.shape[2:], mode='bilinear', align_corners=False)
        skip0_att = self.att1(skip0, y)
        y = torch.cat([y, skip0_att], dim=1)
        y = self.c1(y)

        # Final upsample
        y = F.interpolate(y, size=input_size, mode='bilinear', align_corners=False)
        logits = self.head(y)
        return logits

# --------------------------
# Student wrapper
# --------------------------
class StudentModel(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, dropout=0.2):
        super().__init__()
        self.net = EfficientNetFCN_Attention(in_ch=in_ch, out_ch=out_ch, dropout=dropout)

    def forward(self, x):
        return self.net(x)
