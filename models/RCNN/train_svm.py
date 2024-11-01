from config import TrainSVMOptions
from dataset import Flowers2
from transform import transform

import click
from sklearn import svm
from pathlib import Path
import joblib

@click.command()
@click.option("--comment", type=str, default="", show_default=False)
@click.option("--model_base_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--image_size", type=int, default=224, show_default=True)
def train(**kwargs):
    opts = TrainSVMOptions().parse(kwargs, present=True)

    dataset = Flowers2(
            root=opts.dataset_path,
            image_size=opts.image_size,
            transform=transform(opts.image_size)
        )
    
    [f1,l1,f2,l2] = dataset.svm_load_data()
    
    clf1 = svm.LinearSVC()
    clf1.fit(f1, l1)

    clf2 = svm.LinearSVC()
    clf2.fit(f2, l2)

    joblib.dump(clf1, Path(opts.model_base_path) / "checkpoints/1_svm.pkl")
    joblib.dump(clf2, Path(opts.model_base_path) / "checkpoints/2_svm.pkl")

if __name__ == "__main__":
    train()