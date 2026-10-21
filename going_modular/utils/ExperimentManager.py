import os
import json
import datetime

class ExperimentManager:
    def __init__(self, conf):
        # 1. Tạo đường dẫn gốc: experiments/Tên_Thí_Nghiệm
        # VD: .../experiments/Single_ALBEDO_Shuffle
        self.exp_dir = os.path.join(conf['dataset_dir'], 'experiments', conf['note'])

        # 2. Định nghĩa các thư mục con
        self.ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        self.txt_log_path = os.path.join(self.exp_dir, 'log.txt')
        self.config_path = os.path.join(self.exp_dir, 'config.json')

        # 3. Tạo thư mục
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 4. Lưu Config thành file JSON
        self.save_config(conf)

        # 5. Ghi log khởi tạo
        self.log_text(f"KHOI TAO THI NGHIEM: {conf['note']}")
        self.log_text(f"Luu tru tai: {self.exp_dir}")
        self.log_text(f"Thoi gian: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_text("-" * 50)

    def save_config(self, conf):
        # Chuyển các object không phải json (như device) thành string
        try:
            json_conf = {k: str(v) for k, v in conf.items()}
            with open(self.config_path, 'w') as f:
                json.dump(json_conf, f, indent=4)
        except Exception as e:
            print(f"Warning: Khong the luu config.json ({e})")

    def log_text(self, msg):
        # 1. In ra màn hình console
        print(msg)
        # 2. Ghi vào file log.txt
        with open(self.txt_log_path, 'a') as f:
            f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")

    def log_metrics(self, epoch, metrics):
        # Ghi metrics dạng text rút gọn
        msg = f"Ep {epoch}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log_text(msg)