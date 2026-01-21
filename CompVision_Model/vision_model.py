import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()

        # We load a U-Net with a ResNet-34 encoder.
        # "encoder_weights='imagenet'" downloads the pre-trained knowledge.
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",  # The "Knowledge Base"
            in_channels=n_channels,  # 3 for RGB
            classes=n_classes,  # 1 for Socket Mask
        )

    def forward(self, x):
        return self.model(x)