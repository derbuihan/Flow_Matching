import math

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def get_cifar10_loader(data_dir="data", batch_size=128, num_workers=4):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = datasets.CIFAR10(root=data_dir, download=True, transform=transform)

    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def plot_images(images, path="images/generated.png"):
    images = images.detach().cpu()
    images = (images.clamp(-1, 1) + 1) / 2

    grid = make_grid(images, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class UNet(nn.Module):
    """Small U-Net for 32x32 CIFAR images."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )

        def down_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
            )

        self.time_embedding = SinusoidalTimeEmbedding(128)
        self.time_proj_64 = nn.Linear(128, 64)
        self.time_proj_128 = nn.Linear(128, 128)
        self.time_proj_256 = nn.Linear(128, 256)
        self.time_proj_512 = nn.Linear(128, 512)

        self.encoder1 = conv_block(in_channels, 64)
        self.down1 = down_block(64, 128)
        self.encoder2 = conv_block(128, 128)
        self.down2 = down_block(128, 256)
        self.encoder3 = conv_block(256, 256)
        self.down3 = down_block(256, 512)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, 16, 512))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.up3 = up_block(512, 256)
        self.decoder3 = conv_block(256 + 256, 256)
        self.up2 = up_block(256, 128)
        self.decoder2 = conv_block(128 + 128, 128)
        self.up1 = up_block(128, 64)
        self.decoder1 = conv_block(64 + 64, 64)
        self.output = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x, t):
        t = t.view(t.shape[0], 1)
        temb = self.time_embedding(t)

        x0 = self.encoder1(x)  # (B, 64, 32, 32)
        x0 = x0 + self.time_proj_64(temb)[:, :, None, None]

        x1 = self.encoder2(self.down1(x0))  # (B, 128, 16, 16)
        x1 = x1 + self.time_proj_128(temb)[:, :, None, None]

        x2 = self.encoder3(self.down2(x1))  # (B, 256, 8, 8)
        x2 = x2 + self.time_proj_256(temb)[:, :, None, None]

        x3 = self.down3(x2)  # (B, 512, 4, 4)
        x3 = x3 + self.time_proj_512(temb)[:, :, None, None]

        batch_size, channels, height, width = x3.shape
        x3 = x3.flatten(2).transpose(1, 2)
        x3 = x3 + self.pos_embedding
        x3 = self.transformer(x3)
        x3 = x3.transpose(1, 2).reshape(batch_size, channels, height, width)

        x = self.up3(x3)  # (B, 256, 8, 8)
        x = self.decoder3(torch.cat((x, x2), dim=1))  # (B, 256, 8, 8)

        x = self.up2(x)  # (B, 128, 16, 16)
        x = self.decoder2(torch.cat((x, x1), dim=1))  # (B, 128, 16, 16)

        x = self.up1(x)  # (B, 64, 32, 32)
        x = self.decoder1(torch.cat((x, x0), dim=1))  # (B, 64, 32, 32)

        return self.output(x)  # (B, 3, 32, 32)


def train_model(model, batch_size, num_epochs, device):
    loader = get_cifar10_loader(batch_size=batch_size)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        num_samples = 0
        for images, _ in loader:
            images = images.to(device)
            t = torch.rand(images.shape[0], 1, 1, 1, device=device)
            x_0 = torch.randn_like(images)

            x_t = (1 - t) * x_0 + t * images
            target_velocity = images - x_0
            pred_velocity = model(x_t, t)

            loss = ((pred_velocity - target_velocity) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            num_samples += images.shape[0]

        avg_loss = total_loss / num_samples
        print(f"epoch {epoch:>4}/{num_epochs} | loss {avg_loss:.4f}")

    torch.save(model.state_dict(), f"models/unet_model_{num_epochs}.pt")


if __name__ == "__main__":
    batch_size = 128
    num_epochs = 20
    model = UNet().to(device)
    train_model(model, batch_size, num_epochs, device)

    # model.load_state_dict(torch.load("models/unet_model_10.pt", map_location=device))
    model.eval()

    x_t = torch.randn(16, 3, 32, 32, device=device)
    num_steps = 100
    gamma = 2.0
    s = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    time_grid = torch.sin(math.pi * s / 2)
    with torch.no_grad():
        for t, t_next in zip(time_grid[:-1], time_grid[1:]):
            t_batch = t.expand(16, 1, 1, 1)
            v_t = model(x_t, t_batch)
            delta_t = t_next - t
            x_t = x_t + v_t * delta_t

    plot_images(x_t, path="images/generated.png")

    # loader = get_cifar10_loader(batch_size=16)
    # for images, _ in loader:
    #     plot_images(images, path="images/target.png")
    #     break
