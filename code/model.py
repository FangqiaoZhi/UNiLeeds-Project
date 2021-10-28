import torch
import torch.nn as nn
import torch.nn.functional as F


class _MergeTwiceConvHelper(nn.Module):
    def __init__(self, input, middle, out):
        super(_MergeTwiceConvHelper, self).__init__()
        layers = [
            nn.Conv2d(input, middle, kernel_size=3, padding=0),
            nn.BatchNorm2d(middle),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle, out, kernel_size=3, padding=0),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet(nn.Module):
    def __init__(self, num, num_classes):
        super(UNet, self).__init__()

        self.LeftLayer1 = _MergeTwiceConvHelper(num, 64, 64)
        self.LeftLayer2 = _MergeTwiceConvHelper(64, 128, 128)
        self.LeftLayer3 = _MergeTwiceConvHelper(128, 256, 256)
        self.LeftLayer4 = _MergeTwiceConvHelper(256, 512, 512)

        self.Middle = _MergeTwiceConvHelper(512, 1024, 1024)

        self.RightLayer4_Upsampling = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.RightLayer4 = _MergeTwiceConvHelper(1024, 512, 512)

        self.RightLayer3_Upsampling = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.RightLayer3 = _MergeTwiceConvHelper(512, 256, 256)

        self.RightLayer2_Upsampling = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.RightLayer2 = _MergeTwiceConvHelper(256, 128, 128)

        self.RightLayer1_Upsampling = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.RightLayer1 = _MergeTwiceConvHelper(128, 64, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        left1_ = self.LeftLayer1(x)
        left1_dwn = F.max_pool2d(left1_, kernel_size=2, stride=2)

        left2_ = self.LeftLayer2(left1_dwn)
        left2_dwn = F.max_pool2d(left2_, kernel_size=2, stride=2)

        left3_ = self.LeftLayer3(left2_dwn)
        left3_dwn = F.max_pool2d(left3_, kernel_size=2, stride=2)

        left4_ = self.LeftLayer4(left3_dwn)
        # left4_ = F.dropout(left4_, p=0.2)
        left4_dwn = F.max_pool2d(left4_, kernel_size=2, stride=2)
        
        middle = self.Middle(left4_dwn)
        #print('l1', left1_.shape, left1_dwn.shape, left2_.shape, left2_dwn.shape, left3_.shape, left3_dwn.shape, left4_.shape, left4_dwn.shape, middle.shape)

        right4_upsampling = self.RightLayer4_Upsampling(middle)
        right4_ = self.RightLayer4(UNet.concat(left4_, right4_upsampling))
        # right4_ = F.dropout(right4_, p=0.2)

        right3_upsampling = self.RightLayer3_Upsampling(right4_)
        right3_ = self.RightLayer3(UNet.concat(left3_, right3_upsampling))

        right2_upsampling = self.RightLayer2_Upsampling(right3_)
        right2_ = self.RightLayer2(UNet.concat(left2_, right2_upsampling))
        # right2_ = F.dropout(right2_, p=0.2)

        right1_upsampling = self.RightLayer1_Upsampling(right2_)
        right1_ = self.RightLayer1(UNet.concat(left1_, right1_upsampling))

        final = self.final(right1_)
        #print('right', right4_upsampling.shape, right3_upsampling.shape, right2_upsampling.shape, right1_upsampling.shape, final.shape)
        #print('right_', right4_.shape, right3_.shape, right2_.shape, right1_.shape)
        return final

    @staticmethod
    def concat(bypass, upsampled):
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        if c != 0:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

