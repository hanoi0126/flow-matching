import os
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from dotenv import load_dotenv
from omegaconf import OmegaConf
from scipy import integrate
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from config import MainConfig
from model.unet import Unet

load_dotenv()


def euler_sampler(
    model: Unet,
    shape: tuple[int, int, int, int],
    sample_N: int,
    eps: float,
    device: torch.device,
    condition: bool,
) -> tuple[Tensor, int]:
    model.eval()
    cond = torch.arange(10).repeat(shape[0] // 10).to(device) if condition else None
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        dt = 1.0 / sample_N
        for i in range(sample_N):
            num_t = i / sample_N * (1 - eps) + eps
            t = torch.ones(shape[0], device=device) * num_t
            pred = model(x, t * 999, cond)

            x = x.detach().clone() + pred * dt

        nfe = sample_N
        return x.cpu(), nfe


def rk45_sampler(
    model: Unet,
    shape: tuple[int, int, int, int],
    eps: Tensor,
    device: torch.device,
    condition: bool,
) -> tuple[Tensor, Any]:

    rtol = atol = 1e-05
    model.eval()
    cond = torch.arange(10).repeat(shape[0] // 10).to(device) if condition else None
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model(x, vec_t * 999, cond)

            return to_flattened_numpy(drift)

        solution = integrate.solve_ivp(
            ode_func,
            (eps, 1),
            to_flattened_numpy(x),
            rtol=rtol,
            atol=atol,
            method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)

        return x, nfe


def to_flattened_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x: Tensor, shape: tuple[int]) -> Tensor:
    return torch.from_numpy(x.reshape(shape))


def imshow(img: Tensor, filename: str) -> None:
    img = img * 0.5 + 0.5
    npimg = img.detach().cpu().numpy()
    npimg = np.clip(npimg, 0, 1).transpose(1, 2, 0)
    plt.imshow(npimg)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)


def save_img_grid(img: Tensor, filename: str) -> None:
    img_grid = torchvision.utils.make_grid(img, nrow=10)
    imshow(img_grid, filename)


def eval(
    model: Unet,
    epoch: int,
    method: str,
    device: torch.device,
    eps: float | Tensor,
    condition: bool,
    output_dir: str,
    sample_N: int | None = None,
    batch_size: int = 100,
) -> None:
    if method == "euler":
        if isinstance(eps, Tensor):
            eps = eps.item()
        if sample_N is None:
            assert False, "sample_N must be specified for euler method"
        images, nfe = euler_sampler(
            model,
            shape=(batch_size, 3, 32, 32),
            eps=eps,
            sample_N=sample_N,
            device=device,
            condition=condition,
        )
    elif method == "rk45":
        if isinstance(eps, float):
            eps = torch.tensor(eps).to(device)
        images, nfe = rk45_sampler(
            model,
            shape=(batch_size, 3, 32, 32),
            eps=eps,
            device=device,
            condition=condition,
        )
    save_img_grid(
        images, f"{output_dir}/{method}_epoch_{epoch + 1:04}_nfe_{nfe:04}.png"
    )


@hydra.main(version_base=None, config_path="config", config_name="cifar10_config")
def train(cfg: MainConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.now_dir)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset=train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(
        dim=32,
        channels=3,
        dim_mults=(1, 2, 4),
        condition=cfg.train.condition,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    progress_bar = tqdm(
        total=cfg.train.num_epochs, desc="Training Progress", leave=True
    )

    for epoch in range(cfg.train.num_epochs):
        total_loss = 0.0
        model.train()

        for batch_idx, (batch, cond) in enumerate(dataloader):
            batch = batch.to(device)

            optimizer.zero_grad()

            z0 = torch.randn_like(batch)
            t = (
                torch.rand(batch.shape[0], device=device) * (1 - cfg.train.eps)
                + cfg.train.eps
            )

            t_expand = t.view(-1, 1, 1, 1).repeat(
                1, batch.shape[1], batch.shape[2], batch.shape[3]
            )
            perturbed_data = t_expand * batch + (1 - t_expand) * z0
            target = batch - z0

            score = model(
                perturbed_data,
                t * 999,
                cond.to(device) if cfg.train.condition else None,
            )

            losses = torch.square(score - target)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)

            loss = torch.mean(losses)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                {"batch_loss": loss.item(), "batch_size": batch.size(0)}
            )

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
        wandb.log({"Epoch": epoch + 1, "Loss": total_loss / len(dataloader)})

        progress_bar.set_postfix({"epoch_loss": total_loss / len(dataloader)})
        progress_bar.update(1)

        if epoch < 10 or (epoch + 1) % 10 == 0:
            eval(
                model,
                epoch,
                "euler",
                device,
                eps=cfg.train.eps,
                condition=cfg.train.condition,
                sample_N=1,
                output_dir=cfg.output_dir,
            )
            eval(
                model,
                epoch,
                "euler",
                device,
                eps=cfg.train.eps,
                condition=cfg.train.condition,
                sample_N=2,
                output_dir=cfg.output_dir,
            )
            eval(
                model,
                epoch,
                "euler",
                device,
                eps=cfg.train.eps,
                condition=cfg.train.condition,
                sample_N=10,
                output_dir=cfg.output_dir,
            )
            eval(
                model,
                epoch,
                "rk45",
                device,
                eps=torch.tensor(cfg.train.eps),
                condition=cfg.train.condition,
                output_dir=cfg.output_dir,
            )

        if (epoch + 1) % 100 == 0:
            os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    f"{cfg.output_dir}/checkpoints", f"model_epoch_{epoch + 1:04}.pt"
                ),
            )


if __name__ == "__main__":
    train()
