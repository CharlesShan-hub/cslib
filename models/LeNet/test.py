import click
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from config import TestOptions
from dataset import MNIST
from transform import transform
from model import load_model


@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--num_classes", type=int, default=10, show_default=True)
@click.option("--use_relu", type=bool, default=False, show_default=True)
@click.option("--use_max_pool", type=bool, default=False, show_default=True)
@click.option("--comment", type=str, default="", show_default=False)
def test(
    model_path, dataset_path, batch_size, num_classes, use_relu, use_max_pool, comment
):

    opts = TestOptions().parse(
        {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "use_relu": use_relu,
            "use_max_pool": use_max_pool,
            "comment": comment,
        },
        present=True,
    )

    dataset = MNIST(
        root=opts.dataset_path, train=False, transform=transform, download=True
    )

    dataloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=False)

    model = load_model(opts)

    with torch.no_grad():
        correct = 0
        total = 0
        pbar = tqdm(dataloader, total=len(dataloader))
        for images, labels in pbar:
            images, labels = images.to(opts.device), labels.to(opts.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the model on the {len(dataloader)} test images: {100 * correct / total:.2f}%"
        )


if __name__ == "__main__":
    test()
