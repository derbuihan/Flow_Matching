import itertools
import math
import os
import random
import time

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


def set_seed(seed):
    random.seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar10_loader(data_dir="data", batch_size=128, num_workers=0, seed=42):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    validation_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=validation_transform
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def plot_images(images, path="images/generated.png"):
    images = images.detach().cpu()
    images = (images.clamp(-1, 1) + 1) / 2

    grid = make_grid(images, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_losses(train_losses, validation_losses, path="images/loss_curve.png"):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("velocity MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
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
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
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


def train_model(
    model,
    batch_size,
    num_epochs,
    device,
    lr=2e-4,
    seed=42,
):
    loader, validation_loader = get_cifar10_loader(batch_size=batch_size, seed=seed)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    train_losses = []
    validation_losses = []
    train_start = time.perf_counter()

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
        train_losses.append(avg_loss)

        model.eval()
        validation_total = 0.0
        validation_samples = 0
        with torch.no_grad():
            for images, _ in validation_loader:
                images = images.to(device, non_blocking=True)
                t = torch.rand(images.shape[0], 1, 1, 1, device=device)
                x_0 = torch.randn_like(images)
                x_t = (1 - t) * x_0 + t * images
                target_velocity = images - x_0
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    pred_velocity = model(x_t, t)
                    validation_loss = (
                        (pred_velocity.float() - target_velocity) ** 2
                    ).mean()
                validation_total += validation_loss.item() * images.shape[0]
                validation_samples += images.shape[0]
        avg_validation_loss = validation_total / validation_samples
        validation_losses.append(avg_validation_loss)
        model.train()
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("validation_loss", avg_validation_loss, step=epoch)
        print(
            f"epoch {epoch:>4}/{num_epochs} | "
            f"train {avg_loss:.4f} | validation {avg_validation_loss:.4f}"
        )

    checkpoint_path = f"models/unet_model_{num_epochs}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    mlflow.log_artifact(checkpoint_path)
    return model, train_losses, validation_losses, time.perf_counter() - train_start


@torch.no_grad()
def sample(model, num_samples=16, num_steps=50, device=device, generator=None):
    """Integrate the velocity field from t=0 to t=1 with Heun's method.

    Heun is 2nd order and costs two model calls per step, so `num_steps=50`
    uses the same number of function evaluations as 100 Euler steps.
    """
    model.eval()
    x_t = torch.randn(num_samples, 3, 32, 32, device=device, generator=generator)
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
    num_epochs = int(os.environ.get("NUM_EPOCHS", "20"))
    learning_rate = 2e-4
    seed = 42
    run_name = "upsample-conv"

    set_seed(seed)
    mlflow.set_experiment("CIFAR10-UNet")
    with mlflow.start_run(run_name=run_name):
        model = UNet().to(device)
        model = torch.compile(model)

        mlflow.log_params(
            {
                "model": "UNet-upsample-conv",
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "seed": seed,
                "sampler": "Heun",
                "sampling_steps": 50,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "device": str(device),
                "num_workers": 0,
            }
        )

        model, train_losses, validation_losses, training_seconds = train_model(
            model,
            batch_size,
            num_epochs,
            device,
            lr=learning_rate,
            seed=seed,
        )
        mlflow.log_param("training_seconds", round(training_seconds, 3))

        generated_path = "images/generated.png"
        sampling_generator = torch.Generator(device=device)
        sampling_generator.manual_seed(seed + 1)
        sampling_start = time.perf_counter()
        generated = sample(model, generator=sampling_generator)
        sampling_seconds = time.perf_counter() - sampling_start
        plot_images(generated, path=generated_path)
        mlflow.log_artifact(generated_path)
        mlflow.log_param("sampling_seconds", round(sampling_seconds, 3))

        loss_curve_path = "images/loss_curve.png"
        plot_losses(train_losses, validation_losses, path=loss_curve_path)
        mlflow.log_artifact(loss_curve_path)
