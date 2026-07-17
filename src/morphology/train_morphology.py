"""
src/morphology/train_morphology.py
====================================
Phase 5 · Morphology Classification — EfficientNetB0 Trainer

Trains an EfficientNetB0 transfer-learning classifier on the SMIDS
dataset to classify sperm as Normal / Abnormal.

Expected dataset layout:
    datasets/SMIDS/
    ├── Normal/       ← images of normal sperm
    └── Abnormal/     ← images of abnormal sperm

How to Run
----------
    python -m src.morphology.train_morphology

Output
------
    models/morphology/efficientnet_morphology.pth   ← best checkpoint
    outputs/plots/morphology_training.png
    outputs/plots/morphology_confusion_matrix.png
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dir, load_config, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# EfficientNetB0 Morphology Trainer
# ══════════════════════════════════════════════════════════════

class MorphologyTrainer:
    """
    Transfer-learning trainer using EfficientNetB0 for Normal / Abnormal
    sperm morphology classification.

    Parameters
    ----------
    config : dict, optional
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        morph_cfg = config["morphology"]
        paths_cfg = config["paths"]

        self.img_size     = morph_cfg["image_size"]   # 224
        self.epochs       = morph_cfg["epochs"]
        self.batch_size   = morph_cfg["batch_size"]
        self.lr           = morph_cfg["lr"]
        self.wd           = morph_cfg["weight_decay"]
        self.num_classes  = morph_cfg["num_classes"]
        self.class_names  = morph_cfg["class_names"]
        self.device_str   = morph_cfg["device"]

        self.data_dir     = Path(paths_cfg["datasets"]["smids"])
        self.out_dir      = ensure_dir(Path(paths_cfg["models"]["morphology"]).parent)
        self.ckpt_path    = self.out_dir / "efficientnet_morphology.pth"
        self.plot_dir     = ensure_dir(paths_cfg["outputs"]["plots"])

    # ----------------------------------------------------------

    def _get_device(self):
        """Return torch device."""
        import torch
        if self.device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ----------------------------------------------------------

    def _build_dataloaders(self):
        """
        Construct train / val / test DataLoaders using torchvision
        ImageFolder with ImageNet-normalised transforms.

        Returns
        -------
        dict of DataLoader, dict of dataset sizes
        """
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import random_split, DataLoader

        # ── Augmentation for training ──────────────────────────
        train_tf = transforms.Compose([
            transforms.Resize((self.img_size + 32, self.img_size + 32)),
            transforms.RandomCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # ── Deterministic transform for val / test ─────────────
        val_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        full_dataset = datasets.ImageFolder(str(self.data_dir), transform=train_tf)
        n = len(full_dataset)
        n_val  = int(0.15 * n)
        n_test = int(0.15 * n)
        n_train = n - n_val - n_test

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        # Override transforms for val/test
        val_ds.dataset.transform  = val_tf
        test_ds.dataset.transform = val_tf

        loaders = {
            "train": DataLoader(train_ds, batch_size=self.batch_size,
                                shuffle=True,  num_workers=2, pin_memory=True),
            "val":   DataLoader(val_ds,   batch_size=self.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True),
            "test":  DataLoader(test_ds,  batch_size=self.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True),
        }
        sizes = {"train": n_train, "val": n_val, "test": n_test}

        logger.info("Dataset sizes — train:{} val:{} test:{}", n_train, n_val, n_test)
        logger.info("Classes: {}", full_dataset.classes)

        return loaders, sizes

    # ----------------------------------------------------------

    def _build_model(self, device):
        """
        Build EfficientNetB0 with a custom classification head.

        Returns
        -------
        torch.nn.Module
        """
        import torch
        import torch.nn as nn
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except ImportError:
            # Older torchvision API
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=True)

        # Freeze early layers — fine-tune only the last 3 blocks + head
        for name, param in model.features.named_parameters():
            block_idx = name.split(".")[0]
            if block_idx.isdigit() and int(block_idx) < 5:
                param.requires_grad = False

        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, self.num_classes),
        )

        return model.to(device)

    # ----------------------------------------------------------

    @timer
    def train(self) -> str:
        """
        Full training loop with early stopping.

        Returns
        -------
        str
            Path to saved best checkpoint.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from sklearn.metrics import classification_report, confusion_matrix
        except ImportError:
            logger.error("PyTorch / scikit-learn required: pip install torch torchvision scikit-learn")
            raise

        if not self.data_dir.exists():
            logger.error("SMIDS dataset not found at {}", self.data_dir)
            raise FileNotFoundError(f"Dataset missing: {self.data_dir}")

        device  = self._get_device()
        logger.info("Training on device: {}", device)

        loaders, sizes = self._build_dataloaders()
        model = self._build_model(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=self.lr, weight_decay=self.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        # ── Training loop ──────────────────────────────────────
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc  = 0.0
        patience_cnt  = 0
        patience_max  = 10

        for epoch in range(1, self.epochs + 1):
            for phase in ("train", "val"):
                model.train() if phase == "train" else model.eval()
                running_loss = 0.0
                running_correct = 0

                for inputs, labels in loaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss    = criterion(outputs, labels)
                        preds   = outputs.argmax(dim=1)

                        if phase == "train":
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()

                    running_loss    += loss.item() * inputs.size(0)
                    running_correct += (preds == labels).sum().item()

                epoch_loss = running_loss / sizes[phase]
                epoch_acc  = running_correct / sizes[phase]
                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc)

            scheduler.step()

            val_acc = history["val_acc"][-1]
            logger.info(
                "Epoch {:3d}/{} | Train Loss:{:.4f} Acc:{:.3f} | Val Loss:{:.4f} Acc:{:.3f}",
                epoch, self.epochs,
                history["train_loss"][-1], history["train_acc"][-1],
                history["val_loss"][-1],   val_acc
            )

            # ── Early stopping + best checkpoint ───────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "class_names": self.class_names,
                }, str(self.ckpt_path))
                patience_cnt = 0
                logger.success("  ✓ Best checkpoint saved (val_acc={:.4f})", best_val_acc)
            else:
                patience_cnt += 1
                if patience_cnt >= patience_max:
                    logger.info("Early stopping at epoch {}", epoch)
                    break

        # ── Final evaluation on test set ───────────────────────
        logger.info("Running test evaluation")
        checkpoint = torch.load(str(self.ckpt_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = self._evaluate(model, loaders["test"], device)

        # ── Plots ──────────────────────────────────────────────
        self._plot_training(history)
        self._plot_confusion_matrix(test_metrics["cm"])

        logger.success("Morphology model saved → {}", self.ckpt_path)
        return str(self.ckpt_path)

    # ----------------------------------------------------------

    def _evaluate(self, model, loader, device) -> Dict:
        """
        Evaluate model on a DataLoader, return metrics dict.

        Parameters
        ----------
        model : nn.Module
        loader : DataLoader
        device : torch.device

        Returns
        -------
        dict  with keys: accuracy, precision, recall, f1, cm
        """
        import torch
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix
        )

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds   = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        metrics = {
            "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
            "precision": round(precision_score(all_labels, all_preds, average="weighted",
                                                zero_division=0), 4),
            "recall":    round(recall_score(all_labels, all_preds, average="weighted",
                                             zero_division=0), 4),
            "f1":        round(f1_score(all_labels, all_preds, average="weighted",
                                         zero_division=0), 4),
            "cm":        confusion_matrix(all_labels, all_preds),
        }

        logger.success("Test metrics: Acc={accuracy:.4f} P={precision:.4f} "
                       "R={recall:.4f} F1={f1:.4f}", **metrics)
        return metrics

    # ----------------------------------------------------------

    def _plot_training(self, history: Dict) -> None:
        """Plot loss and accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("EfficientNetB0 Morphology Training", fontsize=14, fontweight="bold")

        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label="Train", color="#e74c3c")
        axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#3498db")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["train_acc"], label="Train", color="#e74c3c")
        axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#3498db")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.plot_dir / "morphology_training.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training plot saved → {}", save_path)

    # ----------------------------------------------------------

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot confusion matrix heatmap."""
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax)
        ax.set_title("Morphology Confusion Matrix", fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        save_path = self.plot_dir / "morphology_confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Confusion matrix saved → {}", save_path)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg     = load_config()
    trainer = MorphologyTrainer(cfg)
    trainer.train()