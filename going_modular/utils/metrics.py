from tabulate import tabulate

class ProgressMeter(object):
    def __init__(self, train_metrics, test_metrics, prefix=""):
        """
        Hiển thị tiến trình training với các giá trị train và test.

        Args:
            train_metrics (dict): Từ điển chứa tên tham số và giá trị của train.
            test_metrics (dict): Từ điển chứa tên tham số và giá trị của test.
            prefix (str): Tiêu đề hoặc tiền tố cho mỗi epoch.
        """
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.prefix = prefix

    
    def display(self):
        print(self.prefix)

        # Tạo danh sách các metrics và sắp xếp theo thứ tự loss in trước auc in sau
        loss_metrics = [
            "loss", "loss_id", "loss_gender", "loss_da_gender", "loss_emotion", "loss_da_emotion",
            "loss_pose", "loss_da_pose", "loss_facial_hair", "loss_da_facial_hair",
            "loss_spectacles", "loss_da_spectacles"
        ]
        auc_metrics = [
            "auc_gender", "auc_spectacles", "auc_facial_hair", "auc_pose",
            "auc_emotion", "auc_id_cosine", "auc_id_euclidean"
        ]

        # Tạo bảng hiển thị từ các metric
        headers = ["Metric", "Train", "Test"]
        rows = []

        # Sắp xếp các metrics theo order loss trước auc
        for metric in loss_metrics + auc_metrics:
            rows.append([metric, self.train_metrics.get(metric, "-"), self.test_metrics.get(metric, "-")])

        # Sử dụng tabulate để tạo bảng
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
        
class ConcatProgressMeter(object):
    def __init__(self, train_metrics, test_metrics, prefix=""):
        """
        Hiển thị tiến trình training với các giá trị train và test.

        Args:
            train_metrics (dict): Từ điển chứa tên tham số và giá trị của train.
            test_metrics (dict): Từ điển chứa tên tham số và giá trị của test.
            prefix (str): Tiêu đề hoặc tiền tố cho mỗi epoch.
        """
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.prefix = prefix

    
    def display(self):
        print(self.prefix)

        # Tạo danh sách các metrics và sắp xếp theo thứ tự loss in trước auc in sau
        loss_metrics = [
            "loss", "loss_id", "loss_gender", "loss_emotion",
            "loss_pose", "loss_facial_hair",
            "loss_spectacles"
        ]
        auc_metrics = [
            "auc_gender", "auc_spectacles", "auc_facial_hair", "auc_pose",
            "auc_emotion", "auc_id_cosine", "auc_id_euclidean"
        ]

        # Tạo bảng hiển thị từ các metric
        headers = ["Metric", "Train", "Test"]
        rows = []

        # Sắp xếp các metrics theo order loss trước auc
        for metric in loss_metrics + auc_metrics:
            rows.append([metric, self.train_metrics.get(metric, "-"), self.test_metrics.get(metric, "-")])

        # Sử dụng tabulate để tạo bảng
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))