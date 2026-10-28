
import os
import json
import datetime
import torch


class ExperimentManager:
    def __init__(self, conf):
        self.conf = conf.copy()
        self.exp_name = conf["note"]

        # 1. Experiment directories
        self.exp_dir = os.path.join(
            conf["dataset_dir"],
            "experiments",
            self.exp_name
        )

        self.ckpt_dir = os.path.join(
            self.exp_dir, "checkpoints"
        )
        self.log_dir = os.path.join(
            self.exp_dir, "logs"
        )

        self.txt_log_path = os.path.join(
            self.exp_dir, "log.txt"
        )
        self.config_path = os.path.join(
            self.exp_dir, "config.json"
        )

        # 2. Create directories
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 3. Start experiment
        self.log_text("=" * 70)
        self.log_text(f"EXPERIMENT: {self.exp_name}")
        self.log_text(f"START TIME: {self._now()}")
        self.log_text(f"OUTPUT DIR: {self.exp_dir}")
        self.log_text("=" * 70)

        self.save_config()

        # 4. Automatically log basic configuration
        self.log_config()

    @staticmethod
    def _now():
        return datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @staticmethod
    def _format_value(value):
        if isinstance(value, torch.device):
            return str(value)
        return str(value)

    def log_text(self, msg):
        """Print to console and append to log.txt."""
        msg = str(msg)
        print(msg)

        with open(self.txt_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{self._now()}] {msg}\n"
            )

    def save_config(self):
        """Save current config as JSON."""
        json_conf = {
            k: self._format_value(v)
            for k, v in self.conf.items()
        }

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(
                json_conf,
                f,
                indent=4,
                ensure_ascii=False
            )

    def update_config(self, **kwargs):
        """Update config and persist it."""
        self.conf.update(kwargs)
        self.save_config()

    def log_config(self):
        self.log_text("[CONFIGURATION]")
        for key, value in self.conf.items():
            self.log_text(f"  {key}: {value}")

    def log_training_start(
        self,
        model_name=None,
        training_mode=None,
        use_sampler=None,
        num_classes=None,
        train_samples=None,
        val_samples=None,
        batch_size=None,
        epochs=None
    ):
        """Log training setup."""
        self.log_text("-" * 70)
        self.log_text("[TRAINING SETUP]")

        info = {
            "Training mode": training_mode,
            "Backbone": model_name,
            "PK Sampler": use_sampler,
            "Num classes": num_classes,
            "Train samples": train_samples,
            "Validation samples": val_samples,
            "Batch size": batch_size,
            "Epochs": epochs,
        }

        for key, value in info.items():
            if value is not None:
                self.log_text(f"{key}: {value}")

        self.log_text("-" * 70)

    def log_metrics(self, epoch, metrics):
        """Log epoch metrics."""
        parts = []

        for key, value in metrics.items():
            try:
                parts.append(f"{key}: {float(value):.4f}")
            except (ValueError, TypeError):
                parts.append(f"{key}: {value}")

        self.log_text(
            f"Epoch {epoch}: " + ", ".join(parts)
        )

    def log_epoch(self, epoch, train_loss=None, val_loss=None,
                  metrics=None, lr=None):
        """Log epoch summary."""
        parts = []

        if train_loss is not None:
            parts.append(f"train_loss: {train_loss:.4f}")

        if val_loss is not None:
            parts.append(f"val_loss: {val_loss:.4f}")

        if lr is not None:
            parts.append(f"lr: {lr:.8g}")

        if metrics:
            for key, value in metrics.items():
                try:
                    parts.append(f"{key}: {float(value):.4f}")
                except (ValueError, TypeError):
                    parts.append(f"{key}: {value}")

        self.log_text(f"Epoch {epoch}: " + ", ".join(parts))

    def log_section(self, title):
        self.log_text("")
        self.log_text("=" * 70)
        self.log_text(title)
        self.log_text("=" * 70)

    def log_training_end(self, message="Training completed"):
        self.log_section(message)
        self.log_text(f"END TIME: {self._now()}")
        self.log_text(f"OUTPUT DIR: {self.exp_dir}")
