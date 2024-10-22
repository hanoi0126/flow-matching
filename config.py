from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    eps: float
    condition: bool


@dataclass
class WandbConfig:
    project: str
    entity: str


@dataclass
class MainConfig:
    wandb: WandbConfig
    train: TrainConfig
    now_dir: str
    output_dir: str
