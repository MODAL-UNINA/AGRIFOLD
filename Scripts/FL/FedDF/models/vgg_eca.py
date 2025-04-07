import torch
import torch.nn as nn
from torchvision import models

class Net(nn.Module):
    """Constructs a ECA module.

    Args:
      channel: Number of channels of the input feature map
      k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(Net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.feature_maps = None

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        self.feature_maps = x * y.expand_as(x)

        return self.feature_maps


class VGG16WithECA(nn.Module):
    def __init__(self, num_classes=10, kernel_size=3, pretrained=True):
        super(VGG16WithECA, self).__init__()
        
        vgg = models.vgg16(pretrained=pretrained)

        for param in vgg.features.parameters():
            param.requires_grad = True

        self.features = nn.Sequential(

            *vgg.features[:5],  # Conv1-Conv2 + ReLU
            Net(kernel_size),

            *vgg.features[5:10],  # Conv3-Conv4 + ReLU
            Net(kernel_size),

            *vgg.features[10:17],  # Conv5-Conv7 + ReLU
            Net(kernel_size),

            *vgg.features[17:24],  # Conv8-Conv10 + ReLU
            Net(kernel_size),

            *vgg.features[24:],  # Conv11-Conv13 + ReLU
            Net(kernel_size),
        )

        # Classifier 
        self.classifier = vgg.classifier
        self.classifier[-1] = nn.Linear(4096, num_classes)


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_last_eca_feature_maps(self):
        return self.features[-1].feature_maps



def vgg_eca(conf):

    model = VGG16WithECA(
        kernel_size=3,
    )
    return model