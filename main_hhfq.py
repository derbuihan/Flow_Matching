import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid


class FFHQDataset(Dataset):
    """FFHQ 128×128 thumbnails"""

    def __init__(self, data_dir, transform=None):
        self.image_paths = sorted(Path(data_dir).glob("*.png"))
        self.transform = transform

        if not self.image_paths:
            raise ValueError(f"No PNG images found in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def get_ffhq_loader(
    data_dir="data/ffhq/thumbnails128x128", batch_size=128, num_workers=8, seed=42
):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    loader = DataLoader(
        FFHQDataset(data_dir, transform),
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
    return loader


def plot_images(images, path="images/ffhq_samples.png"):
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
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=128):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(math.gcd(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, temb):
        h = self.block1(x)
        time_emb = self.time_proj(temb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        time_emb = self.time_proj(temb)[:, :, None, None]
        return h + time_emb + self.conv1(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
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

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Small U-Net for 128x128 FFHQ images."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(128)

        self.encoder1 = ResBlock(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.encoder2 = ResBlock(128, 128)
        self.down2 = DownBlock(128, 256)
        self.encoder3 = ResBlock(256, 256)
        self.down3 = DownBlock(256, 512)
        self.encoder4 = ResBlock(512, 512)
        self.down4 = DownBlock(512, 1024)
        self.encoder5 = ResBlock(1024, 1024)
        self.down5 = DownBlock(1024, 2048)

        self.bottleneck = ResBlock(2048, 2048)

        self.up5 = UpBlock(2048, 1024)
        self.decoder5 = ResBlock(1024 + 1024, 1024)
        self.up4 = UpBlock(1024, 512)
        self.decoder4 = ResBlock(512 + 512, 512)
        self.up3 = UpBlock(512, 256)
        self.decoder3 = ResBlock(256 + 256, 256)
        self.up2 = UpBlock(256, 128)
        self.decoder2 = ResBlock(128 + 128, 128)
        self.up1 = UpBlock(128, 64)
        self.decoder1 = ResBlock(64 + 64, 64)

        self.output = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x, t):
        t = t.view(t.shape[0], 1)

        temb = self.time_embedding(t)

        x0 = self.encoder1(x, temb)
        x1 = self.encoder2(self.down1(x0), temb)
        x2 = self.encoder3(self.down2(x1), temb)
        x3 = self.encoder4(self.down3(x2), temb)
        x4 = self.encoder5(self.down4(x3), temb)
        x5 = self.bottleneck(self.down5(x4), temb)

        x = self.up5(x5)
        x = self.decoder5(torch.cat((x, x4), dim=1), temb)
        x = self.up4(x)
        x = self.decoder4(torch.cat((x, x3), dim=1), temb)
        x = self.up3(x)
        x = self.decoder3(torch.cat((x, x2), dim=1), temb)
        x = self.up2(x)
        x = self.decoder2(torch.cat((x, x1), dim=1), temb)
        x = self.up1(x)
        x = self.decoder1(torch.cat((x, x0), dim=1), temb)

        return self.output(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    loader = get_ffhq_loader()
    model = UNet().to(device)
    model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters())

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            t = torch.rand(images.shape[0], 1, 1, 1, device=device)
            x_0 = torch.randn_like(images)

            x_t = (1 - t) * x_0 + t * images
            target_velocity = images - x_0

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_velocity = model(x_t, t)
                loss = ((pred_velocity - target_velocity) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    for images, _ in loader:
        print(f"images: {images.shape}")
        plot_images(images[:16])
        break
