import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            # nn.GroupNorm(8, cout),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(cout, cout, 3, padding=1),
            # nn.GroupNorm(8, cout),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(cout, cout, 3, padding=1),
            # nn.GroupNorm(8, cout),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(cout, cout, 3, padding=1),
            # nn.GroupNorm(8, cout),
            nn.InstanceNorm2d(cout, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Dropout2d(.2)
        )
        self.res_conv = nn.Conv2d(cin, cout, kernel_size=1, bias=False)

        print(cin, cout)
    def forward(self, x):
        residual = self.res_conv(x)
        out = self.conv(x)
        return out + residual


class ResNetEncoder(nn.Module):
    """ResNet34 encoder modified for 256x256 input"""
    def __init__(self, in_ch=1, pretrained=True):
        super().__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        if in_ch != 3:
            resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False)
        resnet.maxpool = nn.Identity()  # prevent downsampling
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

class FCNBaseline(nn.Module):
    """A minimal FCN with light encoder-decoder and skip connections."""
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.encoder = ResNetEncoder(in_ch=in_ch, pretrained=True)

        self.u2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c4 = ConvBlock(256 + 256, 256)
        self.u1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c5 = ConvBlock(128 + 128, 128)
        # Final upsample to 256x256
        self.u0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c6 = ConvBlock(64 + 64, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        # print("X SHAPE: ", x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        y = self.u2(x4)
        y = torch.cat([y, x3], dim=1)
        y = self.c4(y)
        y = self.u1(y)
        y = torch.cat([y, x2], dim=1)
        y = self.c5(y)
        y = self.u0(y)
        y = torch.cat([y, x0], dim=1)
        y = self.c6(y)
        # logits = self.out(y)
        logits = torch.sigmoid(self.out(y))

        # print("Y SHAPE ", y.shape)
        # print("logits SHAPE ", logits.shape)
        return logits

class StudentModel(nn.Module):
    """Students: replace the backbone with your own architecture."""
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.net = FCNBaseline(in_ch, base_ch)
    def forward(self, x):
        return self.net(x)

# Weight Initialization
# Since you’re mixing pretrained and new layers, consider:
#
# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
# self.apply(init_weights)
