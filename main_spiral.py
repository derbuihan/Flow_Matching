import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def plot_x(x_t, path="images/x_t.png", label="x_t"):
    x_t = x_t.detach().cpu()
    sns.scatterplot(x=x_t[:, 0], y=x_t[:, 1], label=label, s=6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


class FlowModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.W1 = nn.Linear(3, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.W3 = nn.Linear(hidden, 2)

    def forward(self, x_t, t):
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        t = t.reshape(-1, 1)
        x_t_with_t = torch.cat([x_t, t], dim=1)
        y1 = F.relu(self.W1(x_t_with_t))
        y2 = F.relu(self.W2(y1))
        return self.W3(y2)


def train_model(model, batch_size, epochs, device):
    # x_0: noise
    # x_1: real data
    # x_t = (1-t) * x_0 + t * x_1
    # v_t = x_1 - x_0

    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        t = torch.rand(batch_size, 1, device=device)

        theta = torch.rand(batch_size, device=device) * 4 * math.pi
        r = theta / (4 * math.pi)
        x_1 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        x_1 += 0.03 * torch.randn_like(x_1)
        x_0 = torch.randn_like(x_1)

        x_t = (1 - t) * x_0 + t * x_1
        pred_velocity = model(x_t, t)
        target_velocity = x_1 - x_0

        loss = ((pred_velocity - target_velocity) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# x_0: noise
# x_1: real data
# x_t = (1-t) * x_0 + t * x_1
# v_t = x_1 - x_0

model = FlowModel().to(device)


batch_size = 2048
epochs = 100000

model = train_model(model, batch_size, epochs, device)

x_t = torch.randn(batch_size, 2, device=device)
dt = 0.01

model.eval()
with torch.no_grad():
    # for t in torch.arange(0.0, math.pi / 2, dt, device=device):
    #     t = torch.sin(t)
    for t in torch.arange(0.0, 1.0, dt, device=device):
        t_batch = t.expand(batch_size, 1)
        v_t = model(x_t, t_batch)
        x_t = x_t + v_t * dt

plot_x(x_t, path="images/generated.png", label="generated")
