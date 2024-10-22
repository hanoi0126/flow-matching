from dataclasses import dataclass


@dataclass
class Train:
    batch_size: int
    learning_rate: float
    num_epochs: int
    eps: float


@dataclass
class Wandb:
    project: str
    entiry: str


@dataclass
class MainConfig:
    wandb: Wandb = Wandb()
    train: Train = Train()
    now_dir: str
    output_dir: str
