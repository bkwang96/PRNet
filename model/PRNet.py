import torch
from torch import nn
import torch.nn.functional as F
from model.build_contextpath import build_contextpath

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)

        return x

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
    def forward(self, input):

        x = torch.mean(input, 3, keepdim=True)
        x = torch.mean(x, 2, keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, fh):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=1024, out_channels=128, stride=1)
        self.conv1 = nn.Conv2d(128,128,kernel_size=(1,7),padding=(0,3))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(7, 1), padding=(3, 0))
        self.fh=fh

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)

        x = self.convblock1(x)

        for i in range(self.fh-1):
            x[:,:,i+1,:]=x[:,:,i+1,:]+self.conv1(x[:,:,i,:].unsqueeze(2)).squeeze(2)

        return x

class PRNet(torch.nn.Module):
    def __init__(self, context_path):
        super().__init__()

        self.saptial_path = Spatial_path()

        self.context_path = build_contextpath(name=context_path)

        self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.feature_fusion_module = FeatureFusionModule(32)

        self.loc=nn.Conv2d(128,3,kernel_size=3, padding=1)
        self.conf=nn.Conv2d(128,2,kernel_size=3, padding=1)
        self.end = nn.Conv2d(128, 1, kernel_size=3, padding=1)




    def forward(self, image):

        sx = self.saptial_path(image)

        cx1, cx2, tail  = self.context_path(image)

        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        cx2 = self.upsample_4(cx2)
        cx1 = self.upsample_2(cx1)
        cx = torch.cat((cx1, cx2), dim=1)

        x = self.feature_fusion_module(sx, cx)

        out_conf = self.conf(x)
        out_loc = self.loc(x)
        out_end = self.end(x)

        out_conf = F.softmax(out_conf, dim=1)
        out_conf = F.pad(out_conf, (1, 1, 1, 1), "constant", 0)

        return out_conf, out_loc, out_end



