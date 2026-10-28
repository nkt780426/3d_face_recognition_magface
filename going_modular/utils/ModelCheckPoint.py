import torch
import shutil
from termcolor import cprint
import os

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class ModelCheckpoint:
    def __init__(self, output_dir, save_freq=1, verbose=1):
        """
        Args:
            output_dir (str): Thư mục lưu checkpoint.
            save_freq (int): Tần suất lưu checkpoint (ví dụ: save_freq=5 nghĩa là cứ 5 epoch lưu 1 lần).
            verbose (int): Mức độ in log.
        """
        self.output_dir = output_dir
        self.save_freq = save_freq
        self.verbose = verbose

        os.makedirs(self.output_dir, exist_ok=True)
        self.last_path = os.path.join(self.output_dir, "last_model.pth")

    def __call__(self, model, optimizer, epoch, metrics=None, scheduler=None):
        """
        epoch: Epoch hiện tại, đánh số từ 1 (theo convention hiển thị).
        """
        # Chỉ lưu checkpoint nếu đạt tần suất yêu cầu hoặc là epoch cuối (nếu cần)
        if epoch % self.save_freq != 0:
            return

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, self.last_path)

        if self.verbose:
            cprint(f"Saved checkpoint: epoch {epoch}", "cyan")

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None, map_location="cpu", strict=True):
        """
        Resume từ last_model.pth.
        Trả về:
            start_epoch: Epoch tiếp theo cần train (0-based index).
        """
        path = checkpoint_path or self.last_path

        if not os.path.isfile(path):
            if self.verbose:
                cprint(f"No checkpoint found: {path}", "red")
            return 0

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict
        )

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None:
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            elif self.verbose:
                cprint("Warning: No scheduler state in checkpoint.", "yellow")

        saved_epoch_display = checkpoint.get("epoch", 0)

        if self.verbose:
            cprint(f"Loaded checkpoint: {path}", "cyan")
            cprint(f"Resuming training after epoch {saved_epoch_display}", "cyan")

        return saved_epoch_display