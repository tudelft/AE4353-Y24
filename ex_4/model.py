import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                intermediate_channels, in_channels, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class EncoderResNetVAE(nn.Module):
    def __init__(self, latent_dim, extra_layer=False):
        super(EncoderResNetVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            (
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
                if extra_layer
                else nn.Identity()
            ),
            ResBlock(128, 32),
            ResBlock(128, 32),
        )

        self.embedding = nn.LazyLinear(out_features=self.latent_dim)
        self.log_var = nn.LazyLinear(out_features=self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x_flatten = x.view(x.size(0), -1)
        z = self.embedding(x_flatten)
        log_var = self.log_var(x_flatten)

        return z, log_var


class DecoderResNetVAE(nn.Module):
    def __init__(
        self, latent_dim, bottleneck_shape=(4, 4), channel_out=1, extra_layer=False
    ):
        super(DecoderResNetVAE, self).__init__()

        self.latent_dim = latent_dim

        self.bottleneck_shape = bottleneck_shape
        bottleneck_size = torch.prod(torch.tensor(bottleneck_shape)).item()

        self.projection_layer = nn.Linear(self.latent_dim, 128 * bottleneck_size)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1),
            ResBlock(128, 32),
            ResBlock(128, 32),
            nn.ReLU(),
            (
                nn.ConvTranspose2d(
                    128, 128, kernel_size=5, stride=2, padding=1, output_padding=1
                )
                if extra_layer
                else nn.Identity()
            ),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, channel_out, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.projection_layer(z)
        x = x.view(x.size(0), 128, *self.bottleneck_shape)
        x = self.upsampling(x)
        return x
