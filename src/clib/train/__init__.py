import torch
from pathlib import Path
from tqdm import tqdm
from collections import deque
import os
import random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold


class Components:
    def __init__(self, opts) -> None:
        self.opts = opts
        self.epoch_history = 0
        self._set_seed()
        self._set_demo_components()
        self._build_folder()
        self._set_logger()

    def _set_seed(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.seed_worker = seed_worker

        g = torch.Generator()
        g.manual_seed(self.opts.seed)
        self.g = g

        torch.manual_seed(self.opts.seed)  # random seed (only cpu)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.opts.seed)
            torch.cuda.manual_seed_all(self.opts.seed)  # seed all GPUs
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _set_demo_components(self):
        self._model = torch.nn.Linear(256, 120)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.transform = transforms.Compose([])
        empty_dataset = TensorDataset(
            torch.tensor([1, 2, 3, 4, 5])
        )  # torch.utils.data.Dataset()
        self.train_loader = DataLoader(
            dataset=empty_dataset, batch_size=64, shuffle=True
        )
        self.val_loader = DataLoader(dataset=empty_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(
            dataset=empty_dataset, batch_size=64, shuffle=True
        )

    def _build_folder(self):
        assert hasattr(self.opts, "model_base_path")
        if Path(self.opts.model_base_path).exists() == False:
            os.makedirs(self.opts.model_base_path)
        else:
            if list(Path(self.opts.model_base_path).iterdir()):
                raise SystemError(f"{self.opts.model_base_path} should be empty")

    def _set_logger(self):
        self.writer = SummaryWriter(log_dir=self.opts.model_base_path)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if not isinstance(value, torch.nn.Module):
            raise ValueError("model must be torch.nn.modules")
        self._model = value.to(self.opts.device)

    def save(self):
        torch.save(
            self.model.state_dict(), Path(self.opts.model_base_path, "model.pth")
        )
        self.opts.save()


class BaseTrainer(Components):
    def __init__(self, opts, TrainOptions=None, **kwargs):
        if TrainOptions is not None:
            opts = TrainOptions().parse(opts, present=False)
        super().__init__(opts)
        # build_list = ['model','criterion','optimizer','transform','train_loader','val_loader','test_loader']
        # for item in build_list:
        #     value = kwargs[item] if item in kwargs else getattr(self, f'default_{item}')()
        #     if (value is not None):
        #         setattr(self,item,value)

    def adjust_learning_rate(self, factor=0.1):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor

    def train_Holdout(self):
        def train_in_epoch(pbar, epoch):
            running_loss = torch.tensor(0.0).to(self.opts.device)
            batch_index = 0
            for images, labels in pbar:
                images, labels = images.to(self.opts.device), labels.to(
                    self.opts.device
                )
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_index += 1
                running_loss += loss
                pbar.set_description(
                    f"Epoch [{epoch}/{self.opts.epochs if self.opts.epochs != -1 else '∞'}]"
                )
                pbar.set_postfix(loss=(running_loss.item() / batch_index))
            return running_loss / len(self.train_loader)

        def validate_in_epoch():
            val_loss = torch.tensor(0.0).to(self.opts.device)
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.opts.device), labels.to(
                        self.opts.device
                    )
                    outputs = self.model(images)
                    val_loss += self.criterion(outputs, labels).item()
            return val_loss / len(self.val_loader)

        def train_of_epoch():
            recent_losses = deque(maxlen=3)
            epoch = 0
            while True:
                epoch += 1
                pbar = tqdm(self.train_loader, total=len(self.train_loader))
                self.model.train()
                epoch_loss = train_in_epoch(pbar, epoch)
                self.model.eval()
                val_loss = validate_in_epoch()
                print(
                    f"Epoch [{epoch}/{self.opts.epochs if self.opts.epochs != -1 else '∞'}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                self.writer.add_scalar(
                    "Loss/train", epoch_loss, epoch + self.epoch_history
                )
                self.writer.add_scalar("Loss/val", val_loss, epoch + self.epoch_history)
                recent_losses.append(val_loss)
                if len(recent_losses) == 3 and all(
                    x <= y for x, y in zip(recent_losses, list(recent_losses)[1:])
                ):
                    print("Training has converged. Stopping...")
                    self.epoch_history += epoch
                    break
                if self.opts.epochs != -1 and epoch >= self.opts.epochs:
                    break

        def train_of_repeat():
            for i in range(self.opts.repeat):
                print(f"Starting training round {i+1}/{self.opts.repeat}")
                train_of_epoch()
                if i != self.opts.repeat - 1:
                    self.adjust_learning_rate(factor=self.opts.factor)
                    print(
                        f"Adjusted learning rate to {self.optimizer.param_groups[0]['lr']:.6f} for this round."
                    )

        train_of_repeat()

    def train_K_fold(self):
        self.skf = StratifiedKFold(n_splits=self.opts.fold_num)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images, labels = images.to(self.opts.device), labels.to(
                    self.opts.device
                )
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                f"Accuracy of the model on the {len(self.test_loader)} test images: {100 * correct / total:.2f}%"
            )

    def train(self):
        self.opts.presentParameters()
        torch.manual_seed(self.opts.seed)
        if self.opts.train_mode == "Holdout":
            self.train_Holdout()
        elif self.opts.train_mode == "K-fold":
            self.train_K_fold()
        else:
            self.train_Holdout()
        self.test()
        self.save()
