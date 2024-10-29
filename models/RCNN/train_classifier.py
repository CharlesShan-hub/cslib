import click
from config import TrainOptions
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import AlexNet
from dataset import Flowers17
from transform import transform
from clib.train import ClassifyTrainer

@click.command()
@click.option("--model_base_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--pre_trained", type=bool, default=True, show_default=True)
@click.option("--pre_trained_path", type=click.Path(exists=True), required=True)
@click.option("--pre_trained_url", type=str, required=True)
@click.option("--num_classes", type=int, default=17, show_default=True)
@click.option("--image_size", type=int, default=224, show_default=True)
@click.option("--train_mode", type=str, default="Holdout", show_default=False)
@click.option("--seed", type=int, default=42, show_default=True, required=False)
@click.option("--epochs", type=int, default=2, show_default=True, required=False)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--lr", type=float, default=0.03, show_default=True, required=False)
@click.option("--factor", type=float, default=0.1, show_default=True, required=False)
@click.option("--repeat", type=int, default=2, show_default=True, required=False)
@click.option("--val", type=float, default=0.2, show_default=True, required=False)
@click.option("--comment", type=str, default="", show_default=False)
def train(
    model_base_path,
    dataset_path,
    pre_trained,
    pre_trained_path,
    pre_trained_url,
    num_classes,
    image_size,
    train_mode,
    seed,
    epochs,
    batch_size,
    lr,
    factor,
    repeat,
    val,
    comment,):

    opts = TrainOptions().parse(
        {
            "model_base_path": model_base_path,
            "dataset_path": dataset_path,
            "pre_trained":pre_trained,
            "pre_trained_url":pre_trained_url,
            "pre_trained_path":pre_trained_path,
            "num_classes": num_classes,
            "image_size": image_size,
            "train_mode": train_mode,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "factor": factor,
            "repeat": repeat,
            "val": val,
            "comment": comment,
        }
    )

    trainer = ClassifyTrainer(opts)
    trainer.model = AlexNet(
        num_classes=opts.num_classes,
        classify=True,
        fine_tuning=False
    )
    trainer.criterion = nn.CrossEntropyLoss()
    trainer.optimizer = optim.Adam(params=trainer.model.parameters(), lr=opts.lr)
    trainer.transform = transform(opts.image_size)
    train_dataset = Flowers17(
        root=opts.dataset_path, split="train", download=True, transform=trainer.transform
    )
    val_dataset = Flowers17(
        root=opts.dataset_path, split="val", download=True, transform=trainer.transform
    )
    test_dataset = Flowers17(
        root=opts.dataset_path, split="test", download=True, transform=trainer.transform
    )
    trainer.train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        worker_init_fn=trainer.seed_worker,
        generator=trainer.g,
    )
    trainer.val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        worker_init_fn=trainer.seed_worker,
        generator=trainer.g,
    )
    trainer.test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        worker_init_fn=trainer.seed_worker,
        generator=trainer.g,
    )
    if opts.pre_trained:
        trainer.model.init_weights(
            pre_trained_path=opts.pre_trained_path,
            pre_trained_url=opts.pre_trained_url
        )
    trainer.train()
    

if __name__ == "__main__":
    train()