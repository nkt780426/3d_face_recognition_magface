import os
import torch
from termcolor import cprint

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class MultiMetricEarlyStopping:
    def __init__(
        self,
        min_delta=0,
        patience=0,
        verbose=0,
        mode='min',
        monitor_keys=None,
        start_from_epoch=0,
        save_dir=None
    ):
        """
        PyTorch Multi-Metric EarlyStopping

        Args:
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            verbose (int): Verbosity mode. >0 will print logs.
            mode (str): One of {'min', 'max'}.
            monitor_keys (list): List of metric keys to monitor.
            save_dir (str): Directory to save the best weights for each metric.
        """
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.monitor_keys = monitor_keys or []
        self.start_from_epoch = start_from_epoch
        self.save_dir = save_dir

        if mode not in ['min', 'max']:
            raise ValueError(f"Invalid mode '{mode}', must be 'min' or 'max'.")

        self.monitor_ops = {}
        self.best_values = {}
        self.counters = {}
        self.early_stop = False

        # Initialize tracking for each metric
        for key in self.monitor_keys:
            if mode == 'min':
                self.monitor_ops[key] = lambda current, best: current < best - min_delta
                self.best_values[key] = float('inf')
            elif mode == 'max':
                self.monitor_ops[key] = lambda current, best: current > best + min_delta
                self.best_values[key] = -float('inf')
            self.counters[key] = 0

        # Create save directory if specified
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_info_file = os.path.join(self.save_dir, 'best_epoch.yaml')

    def __call__(self, current_values, model, epoch):
        if epoch < self.start_from_epoch:
            return
        
        if isinstance(current_values, list):
            current_values = dict(zip(self.monitor_keys, current_values))

        all_metrics_stalled = True

        for key in self.monitor_keys:
            current_value = current_values.get(key)
            if current_value is None:
                continue

            # Check for improvement
            if self.monitor_ops[key](current_value, self.best_values[key]):
                self.best_values[key] = current_value
                self.counters[key] = 0
                all_metrics_stalled = False  # Có metric cải thiện

                # Save the best weights
                if self.save_dir:
                    # Remove existing checkpoint for the key
                    existing_files = [f for f in os.listdir(self.save_dir) if f.startswith(f"best_{key}_")]
                    for file in existing_files:
                        os.remove(os.path.join(self.save_dir, file))
                            
                    save_path = os.path.join(self.save_dir, f"best_{key}_{epoch}.pth")
                    torch.save(model.state_dict(), save_path)
                    # self.__update_best_epoch_info(key, epoch)
                    if self.verbose:
                        cprint(f"\tSaved best model weights for '{key}' at epoch {epoch} to '{save_path}'", 'green')
            else:
                self.counters[key] += 1
                if self.verbose:
                    cprint(f"\tEpoch {epoch}: EarlyStopping counter for '{key}': {self.counters[key]}/{self.patience}", 'light_red')

            # Check if patience exceeded for this metric
            if self.counters[key] < self.patience:
                all_metrics_stalled = False  # Có metric chưa đạt ngưỡng patience

        if all_metrics_stalled:
            if self.verbose:
                cprint(f"\tEarly stopping triggered at epoch {epoch} as all metrics exceeded patience.", 'red')
            self.early_stop = True