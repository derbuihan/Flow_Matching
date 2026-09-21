import itertools
import math

import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def get_cifar10_loader(data_dir="data", batch_size=128, num_workers=4):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
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
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
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
        # Each layer produces FiLM scale (gamma) and shift (beta).
        self.time_film_64 = nn.Linear(128, 64 * 2)
        self.time_film_128 = nn.Linear(128, 128 * 2)
        self.time_film_256 = nn.Linear(128, 256 * 2)
        self.time_film_512 = nn.Linear(128, 512 * 2)

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
        gamma, beta = self.time_film_64(temb).chunk(2, dim=1)
        x0 = x0 * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        x1 = self.encoder2(self.down1(x0))  # (B, 128, 16, 16)
        gamma, beta = self.time_film_128(temb).chunk(2, dim=1)
        x1 = x1 * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        x2 = self.encoder3(self.down2(x1))  # (B, 256, 8, 8)
        gamma, beta = self.time_film_256(temb).chunk(2, dim=1)
        x2 = x2 * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        x3 = self.down3(x2)  # (B, 512, 4, 4)
        gamma, beta = self.time_film_512(temb).chunk(2, dim=1)
        x3 = x3 * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

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


def train_model(
    model,
    batch_size,
    num_epochs,
    device,
    lr=2e-4,
):
    loader = get_cifar10_loader(batch_size=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        num_samples = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            t = torch.rand(images.shape[0], 1, 1, 1, device=device)
            x_0 = torch.randn_like(images)

            x_t = (1 - t) * x_0 + t * images
            target_velocity = images - x_0

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_velocity = model(x_t, t)
                loss = ((pred_velocity.float() - target_velocity) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            num_samples += images.shape[0]

        avg_loss = total_loss / num_samples
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"epoch {epoch:>4}/{num_epochs} | loss {avg_loss:.4f}")

    checkpoint_path = f"models/unet_model_{num_epochs}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    mlflow.log_artifact(checkpoint_path)
    return model


@torch.no_grad()
def sample(model, num_samples=16, num_steps=50, device=device):
    """Integrate the velocity field from t=0 to t=1 with Heun's method.

    Heun is 2nd order and costs two model calls per step, so `num_steps=50`
    uses the same number of function evaluations as 100 Euler steps.
    """
    model.eval()
    x_t = torch.randn(num_samples, 3, 32, 32, device=device)
    time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for t, t_next in itertools.pairwise(time_grid):
        delta_t = t_next - t
        v_t = model(x_t, t.expand(num_samples, 1, 1, 1))
        x_euler = x_t + v_t * delta_t
        v_next = model(x_euler, t_next.expand(num_samples, 1, 1, 1))
        x_t = x_t + delta_t * 0.5 * (v_t + v_next)

    return x_t


if __name__ == "__main__":
    batch_size = 128
    num_epochs = 20
    learning_rate = 2e-4
    run_name = "film"

    mlflow.set_experiment("CIFAR10-UNet")
    with mlflow.start_run(run_name=run_name):
        model = UNet().to(device)
        model = torch.compile(model)

        mlflow.log_params(
            {
                "model": "UNet-FiLM",
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "sampler": "Heun",
                "sampling_steps": 50,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "device": str(device),
            }
        )

        model = train_model(
            model,
            batch_size,
            num_epochs,
            device,
            lr=learning_rate,
        )

        generated_path = "images/generated.png"
        plot_images(sample(model), path=generated_path)
        mlflow.log_artifact(generated_path)
