import numpy as np
import albumentations as A
import cv2

# Cải tiến code anh Lâm
class GaussianNoise(A.ImageOnlyTransform):
    
    def __init__(self, mean=0, std=0.1, p =1.0):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def __call__(self, **kwargs):
        row, col, ch = kwargs['image'].shape
        # Tính toán kích thước của vùng ảnh sẽ nhận nhiễu
        min_size = int(row * col * 0.1)
        max_size = int(row * col * 0.25)
        area_size = np.random.randint(min_size, max_size)

        # Tạo ma trận mask để chỉ định vùng của ảnh sẽ nhận nhiễu
        mask = np.zeros((row, col), dtype=np.float32)
        x = np.random.randint(0, col)
        y = np.random.randint(0, row)
        x_end = min(x + int(np.sqrt(area_size)), col)
        y_end = min(y + int(np.sqrt(area_size)), row)
        mask[y:y_end, x:x_end] = 1
        
        # Tạo ma trận nhiễu Gaussian có std ngẫu nhiên
        std = np.random.uniform(0, 0.1)
        gauss = np.random.normal(self.mean, std, (row, col, ch)).astype(np.float32)
        
        # Tiến hành transform trên tất cả các ảnh (chính và bổ sung)
        result = {}

        # Lặp qua tất cả các target (ảnh chính và các ảnh bổ sung)
        for key, image in kwargs.items():
            # Áp dụng nhiễu vào phần của ảnh được chỉ định bởi mask
            noisy_img = (image + gauss * mask[:, :, np.newaxis]).astype(np.float32)
            noisy_img = np.clip(noisy_img, 0, 1)
            # Áp dụng biến đổi cho mỗi ảnh với cùng một giá trị random_scale và tọa độ x, y
            result[key] = noisy_img
            
        return result
    
class RandomResizedCropRect(A.ImageOnlyTransform):
    def __init__(self, size, scale=(0.7, 1.0), p = 1.0):
        super().__init__(p)  # Khởi tạo base class với p là xác suất
        self.size = size
        self.scale = scale
    
    def apply(self, img, random_scale, x, y):
        # Lấy kích thước ảnh
        img_height, img_width, _ = img.shape

        # Tính toán kích thước và tọa độ cho crop
        width = int(img_width * random_scale)
        height = int(img_height * random_scale)

        # Crop ảnh
        img = img[y:y+height, x:x+width]

        # Resize ảnh về kích thước mong muốn
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        return img
    
    def __call__(self, **kwargs):
        # Tạo giá trị ngẫu nhiên chung cho tất cả các ảnh
        random_scale = np.random.uniform(*self.scale)
        img_height, img_width, _ = kwargs['image'].shape  # Lấy kích thước của ảnh chính
        x = np.random.randint(0, img_width - int(img_width * random_scale) + 1)
        y = np.random.randint(0, img_height - int(img_height * random_scale) + 1)
        
        # Tiến hành transform trên tất cả các ảnh (chính và bổ sung)
        result = {}

        # Lặp qua tất cả các target (ảnh chính và các ảnh bổ sung)
        for key, image in kwargs.items():
            # Áp dụng biến đổi cho mỗi ảnh với cùng một giá trị random_scale và tọa độ x, y
            result[key] = self.apply(image, random_scale=random_scale, x=x, y=y)
        
        return result