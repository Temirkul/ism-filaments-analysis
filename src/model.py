import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, apply_batch_norm: bool):
        super().__init__()

        self.apply_batch_norm = apply_batch_norm
        self.batch_norm = (
            nn.BatchNorm2d(num_features=out_channels) if self.apply_batch_norm else None
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # same (not valid) convolutions are used, i.e., image size doesn't change

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=False)
        if self.apply_batch_norm:
            x = self.batch_norm(x)
        x = F.relu(self.conv2(x), inplace=False)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes: int, apply_batch_norm: bool):
        super().__init__()

        self.apply_batch_norm = apply_batch_norm
        self.batch_norms = (
            nn.ModuleList(
                [nn.BatchNorm2d(num_features=i) for i in [1, 64, 128, 256, 512, 1024]]
            )
            if self.apply_batch_norm
            else None
        )

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.up_convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=2 ** (10 - i),
                    out_channels=2 ** (9 - i),
                    kernel_size=(2, 2),
                    stride=(2, 2),
                )
                for i in range(4)
            ]
        )

        self.double_convs = nn.ModuleList(
            [
                DoubleConv(
                    in_channels=1,
                    out_channels=64,
                    apply_batch_norm=self.apply_batch_norm,
                )
            ]
            + [
                DoubleConv(
                    in_channels=2 ** (6 + i),
                    out_channels=2 ** (7 + i),
                    apply_batch_norm=self.apply_batch_norm,
                )
                for i in range(4)
            ]
            + [
                DoubleConv(
                    in_channels=2 ** (10 - i),
                    out_channels=2 ** (9 - i),
                    apply_batch_norm=self.apply_batch_norm,
                )
                for i in range(4)
            ]
        )

        self.final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder part
        if self.apply_batch_norm:
            x = self.batch_norms[0](x)

        skip_connections = (
            []
        )  # really important to define this within the forward pass, not outside it in the __init__ method as self.skip_connections, because then
        for i in range(4):  # values from previous batches will be stored as well
            x = self.double_convs[i](x)
            skip_connections.append(x)
            x = self.pool(x)
            if self.apply_batch_norm:
                x = self.batch_norms[i + 1](x)

        x = self.double_convs[i + 1](x)  # bottleneck

        # decoder part
        for i in range(4):
            x = self.up_convs[i](x)
            x = torch.cat(
                (skip_connections[3 - i], x), dim=1
            )  # concatenate along the channel dimension
            if self.apply_batch_norm:
                x = self.batch_norms[5 - i](x)
            x = self.double_convs[5 + i](x)

        if self.apply_batch_norm:
            x = self.batch_norms[1](x)

        x = self.final_conv(x)

        return x


def main():
    unet = UNet(num_classes=181, apply_batch_norm=False)
    summary(
        unet,
        input_size=(16, 1, 256, 256),
        device="cpu",
        col_names=("input_size", "output_size", "num_params"),
    )

    print("\n\n")

    unet_bn = UNet(num_classes=181, apply_batch_norm=True)
    summary(
        unet_bn,
        input_size=(16, 1, 256, 256),
        device="cpu",
        col_names=("input_size", "output_size", "num_params"),
    )


if __name__ == "__main__":
    main()
