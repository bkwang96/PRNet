import torch
from torchvision import models

class resnet18(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)


    def forward(self, input):
        x = self.features.conv1(input)
        x = self.features.relu(self.features.bn1(x))
        x = self.features.maxpool(x)
        feature1 = self.features.layer1(x)             # 1 / 4
        feature2 = self.features.layer2(feature1)      # 1 / 8
        feature3 = self.features.layer3(feature2)      # 1 / 16
        feature4 = self.features.layer4(feature3)      # 1 / 32
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail



def build_contextpath(name):
    model = {
        'resnet18': resnet18(pretrained=False),
    }
    return model[name]

