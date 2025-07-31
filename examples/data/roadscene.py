import click
from pathlib import Path
from cslib.datasets.fusion import RoadScene

default_dataset_root_path = "/Volumes/Charles/data/vision/torchvision"

@click.command()
@click.option("--dataset-path", default=Path(default_dataset_root_path), type=Path, required=True)
def main(dataset_path: Path):
    dataset = RoadScene(
        dataset_path, 
        download=True, #(default)
        proxy='http://127.0.0.1:7897',
    )
    print(len(dataset)) # 221

if __name__ == '__main__':
    main()