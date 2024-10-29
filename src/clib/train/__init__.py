from typing import Optional
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
from sklearn.model_selection import StratifiedKFold


class Components:
    def __init__(self, opts):
        self.opts = opts
        self._set_seed()
        self._build_folder()
        self._set_components()
        
    def _set_seed(self):
        """
        * Note: The seed_worker function should be used as the 
                worker_init_fn argument when creating a DataLoader
        * Example usage:
        >>> trainer = ClassifyTrainer(opts)
        >>> trainer.train_loader = DataLoader(
        ...     dataset=train_dataset,
        ...     batch_size=opts.batch_size,
        ...     shuffle=True,
        ...     worker_init_fn=trainer.seed_worker,
        ...     generator=trainer.g
        ... )
        """
        # Set the random seed for CPU operations
        torch.manual_seed(self.opts.seed)

        # If CUDA is available, set the random seed for GPU operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.opts.seed)  # Set the seed for the current GPU
            torch.cuda.manual_seed_all(self.opts.seed)  # Set the seed for all GPUs
            torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN algorithms
            torch.backends.cudnn.benchmark = False  # Disable benchmarking to maintain deterministic behavior

        # Set the random seed for NumPy and Python's random module
        np.random.seed(self.opts.seed)
        random.seed(self.opts.seed)

        # Set the seed for DataLoader worker processes
        def seed_worker(worker_id):
            # Derive a worker seed from the initial seed and the worker_id
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        self.seed_worker = seed_worker

        # Set the Generator for shuffle in DataLoader
        g = torch.Generator()
        g.manual_seed(self.opts.seed)
        self.g = g

    def _set_components(self):
        """
        * Note: All Optional components need to reassign before train.
        * Example usage:
        >>> trainer = ClassifyTrainer(opts)
        >>> trainer.model = AlexNet(
        ...     num_classes=opts.num_classes,
        ...     classify=True,
        ...     fine_tuning=False
        ... )
        ... ... 
        """
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler|torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        self.criterion: Optional[torch.nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.val_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None
        self.loss: Optional[torch.Tensor] = None
        self.writer = SummaryWriter(log_dir=self.opts.model_base_path)

    def _build_folder(self):
        """
        * Note 1: The function ensures save results in a new folder. 
        * Note 2: You should use with shell file.
        * Example
        >>> RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"
        >>> NAME=$(date +'%Y_%m_%d_%H_%M')
        >>> mkdir -p "${RES_PATH}/${NAME}"
        """
        assert hasattr(self.opts, "model_base_path")
        assert Path(self.opts.model_base_path).exists()
        if list(Path(self.opts.model_base_path).iterdir()):
            raise SystemError(f"{self.opts.model_base_path} should be empty")

    def save_pth(self):
        """
        Save pth(without network) and opts
        """
        assert self.model is not None
        torch.save(
            self.model.state_dict(), 
            Path(self.opts.model_base_path, "model.pth")
        )
        self.opts.save()
    
    def save_checkpoint(self,epoch):
        """
        TODO : Save ckpt
        """
        assert self.model is not None
        assert self.optimizer is not None
        assert self.loss is not None
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss.item()
        }
        torch.save(checkpoint, Path(self.opts.model_base_path, f"/ckpt/{epoch}.ckpt"))
        


class BaseTrainer(Components):
    def __init__(self, opts):
        super().__init__(opts)

    def adjust_learning_rate(self, factor=0.1):
        assert self.optimizer is not None
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor
    
    def test(self):
        raise RuntimeError("You should implement in subclass")
    
    def holdout(self):
        self.epoch_history: int = 0
        assert self.optimizer is not None 
        for i in range(self.opts.repeat):
            print(f"Starting training round {i+1}/{self.opts.repeat}")
            self.holdout_epoch()
            if i != self.opts.repeat - 1:
                self.adjust_learning_rate(factor=self.opts.factor)
                print(f"Adjusted lr to {self.optimizer.param_groups[0]['lr']:.6f} for this round.")
    
    def holdout_epoch(self):
        assert self.train_loader is not None
        assert self.model is not None
        recent_losses = deque(maxlen=3)
        epoch = 0
        while True:
            epoch += 1
            pbar = tqdm(self.train_loader, total=len(self.train_loader))
            self.model.train()
            train_loss = self.holdout_epoch_train(pbar, epoch)
            self.model.eval()
            val_loss = self.holdout_epoch_validate(epoch)
            recent_losses.append(val_loss)
            if len(recent_losses) == 3 and all(
                x <= y for x, y in zip(recent_losses, list(recent_losses)[1:])
            ):
                print("Training has converged. Stopping...")
                self.epoch_history += epoch
                break
            if self.opts.epochs != -1 and epoch >= self.opts.epochs:
                break

    def holdout_epoch_train(self, pbar, epoch):
        raise RuntimeError("You should implement in subclass")

    def holdout_epoch_validate(self):
        raise RuntimeError("You should implement in subclass")

    def k_fold(self):
        self.skf = StratifiedKFold(n_splits=self.opts.fold_num)

    def train(self):
        self.opts.presentParameters()
        if self.opts.train_mode == "Holdout":
            self.holdout()
        elif self.opts.train_mode == "K-fold":
            self.k_fold()
        else:
            self.holdout()
        self.test()
        self.save()


class ClassifyTrainer(BaseTrainer):
    def __init__(self, opts):
        super().__init__(opts)

    def holdout_epoch_train(self, pbar, epoch):
        assert self.optimizer is not None
        assert self.model is not None
        assert self.criterion is not None
        assert self.train_loader is not None
        running_loss = torch.tensor(0.0).to(self.opts.device)
        batch_index = 0
        correct = 0
        total = 0
        for images, labels in pbar:
            images = images.to(self.opts.device)
            labels = labels.to(self.opts.device)
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
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = correct / total
        self.loss = loss
        breakpoint()

        self.writer.add_scalar(
            tag = "Loss/train", 
            scalar_value = train_loss, 
            global_step = epoch + self.epoch_history
        )
        self.writer.add_scalar(
            tag = "Accuracy/train", 
            scalar_value = train_accuracy, 
            global_step = epoch + self.epoch_history
        )

        print(f"Epoch [{epoch}/{self.opts.epochs if self.opts.epochs != -1 else '∞'}]", \
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        return train_loss

    def holdout_epoch_validate(self,epoch):
        assert self.model is not None
        assert self.criterion is not None
        assert self.val_loader is not None
        running_loss = torch.tensor(0.0).to(self.opts.device)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.val_loader:
                images = images.to(self.opts.device)
                labels = labels.to(self.opts.device)
                outputs = self.model(images)
                running_loss += self.criterion(outputs, labels).item()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_accuracy = correct / total
    
        self.writer.add_scalar(
            tag="Loss/val", 
            scalar_value=val_loss, 
            global_step=epoch + self.epoch_history
        )
        self.writer.add_scalar(
            tag = "Accuracy/val", 
            scalar_value = val_accuracy, 
            global_step = epoch + self.epoch_history
        )

        print(f"Epoch [{epoch}/{self.opts.epochs if self.opts.epochs != -1 else '∞'}]", \
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return val_loss
    
    def test(self):
        assert self.model is not None
        assert self.test_loader is not None
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.opts.device)
                labels = labels.to(self.opts.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                f"Accuracy of the model on the {len(self.test_loader)} test images: {100 * correct / total:.2f}%"
            )