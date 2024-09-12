import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


class VarIOResNet(nn.Module):
    def __init__(self, channel_in, channel_out, last_layer_bias=True):
        super().__init__()
        self.resnet = resnet18(weights=None)

        # Change the first layer to accept the number of channels we want
        self.resnet.conv1 = nn.Conv2d(
            channel_in, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Change the last layer to output number of channels we want
        self.resnet.fc = nn.Linear(512, channel_out, bias=last_layer_bias)

    def forward(self, x):
        return self.resnet(x)


class VanillaCNN(nn.Module):
    """
    A vanilla convolutional neural network model.
    with the following architecture:
    - Three convolutional layers each with 3x3 kernels, stride=1 w/ ReLU activation followed by a max pooling layer with 2x2 kernels.
    - The first layer has 16 output channels, the second has 32 output channels, and the third has 64 output channels.
    - Flatten the output of the last convolutional layer.
    - Two fully connected layers with ReLU activation with 128 hidden units and 64 hidden units.
    - One fully connected layer with linear activation as the output layer.

    Args:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        last_layer_bias (bool, optional): Whether to include bias in the last layer. Defaults to True.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        fc1 (nn.LazyLinear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.

    Methods:
        forward(x): Forward pass of the model.

    """

    def __init__(self, channel_in, channel_out, last_layer_bias=True):
        super().__init__()
        # TODO: Implement the model architecture
        self.dummy_param = nn.Parameter(torch.empty(1))
        pass

    def forward(self, x):
        # TODO: Implement the forward pass
        pass


def get_representation_layer(representation):
    assert representation in ["raw", "IQU", "DOP+AOP", "IQU+DOP+AOP"]
    if representation == "raw":
        channel_representation = 4
        representation_layer = nn.Identity()
    elif representation == "IQU":
        channel_representation = 3

        class PolCh2IQU(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # TODO: Implement the conversion from polarized intensity to IQU
                pass

        representation_layer = PolCh2IQU()
    elif representation == "DOP+AOP":
        channel_representation = 2

        class PolCh2DOPAOP(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # TODO: Implement the conversion from polarized intensity to DOP and AOP
                pass

        representation_layer = PolCh2DOPAOP()
    elif representation == "IQU+DOP+AOP":
        channel_representation = 5

        class PolCh2IQUDOPAOP(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # TODO: Implement the conversion from polarized intensity to IQU, DOP and AOP
                pass

        representation_layer = PolCh2IQUDOPAOP()

    return channel_representation, representation_layer


def get_readout_layer(readout, **kwargs):
    assert readout in ["angle", "vector"]
    if readout == "angle":
        out_features = 1

        def get_angle(x):
            return torch.rad2deg(x)

    elif readout == "vector":
        out_features = 2

        def get_angle(x):
            return torch.remainder(torch.rad2deg(torch.atan2(x[:, 1], x[:, 0])), 360)

    return out_features, get_angle


class PolarSunNet(nn.Module):
    def __init__(self, backbone, representation, readout):
        super().__init__()
        channel_representation, self.representation_layer = representation

        out_features, self.get_angle = readout

        self.backbone = backbone(channel_representation, out_features)

    def forward(self, x):
        x = self.representation_layer(x)
        return self.backbone(x)

    def estimate_angle(self, x):
        return self.get_angle(self.forward(x))


if __name__ == "__main__":
    # model = VanillaCNN(4, 2)
    # print(model)
    print(get_readout_layer("classification"))
