import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import cv2, os
from pathlib import Path
from typing import Tuple
import random
import pandas as pd
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
random.seed(seed)

class CustomPhotometricDataset(Dataset):
    
    def __init__(self, dataset_dir: str, transform, type: str = 'normalmap', train: bool = True):
        '''
            type = ['normalmap', 'depthmap', 'albedo']
        '''
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.type = type.lower()
        
        # 1. Xác định file CSV dựa vào tham số train
        split = 'train' if train else 'val' # Hoặc 'test' tùy ý bạn
        self.metadata_path = self.dataset_dir / f"{split}_set.csv"
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metadata tại: {self.metadata_path}")
            
        # Đọc file CSV MỘT LẦN DUY NHẤT và lưu vào self.df
        self.df = pd.read_csv(self.metadata_path)
        
        # 2. Ánh xạ type sang tên file ảnh mới
        filename_mapping = {
            'normalmap': 'normalmap_crop.png',
            'depthmap': 'depthmap_crop.png',
            'albedo': 'albedo_crop.png'
        }
        
        if self.type not in filename_mapping:
            raise ValueError(f"Sai trường type: {type}")
            
        target_filename = filename_mapping[self.type]
        
        # 3. Tạo danh sách self.paths từ file CSV
        # Cấu trúc: dataset_dir / id / session / filename
        self.paths = []
        for _, row in self.df.iterrows():
            img_path = self.dataset_dir / str(row['id']) / str(row['session']) / target_filename
            self.paths.append(img_path)
            
        # 4. Lấy danh sách classes từ cột 'id' trong CSV
        self.classes = sorted(self.df['id'].astype(str).unique().tolist())
        
        self.weightclass = self.__calculate_weight_class()
        
    def __len__(self):
        return len(self.paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        image_path = self.paths[index]

        numpy_image = self.__load_numpy_image(image_path)
        # === 1. CHUẨN HÓA VỀ [0, 1] VÀ ĐỔI SANG FLOAT32 ===
        # Vì ảnh đang ở uint8 (0-255), chia 255.0 để đưa về [0.0, 1.0]
        numpy_image = numpy_image.astype(np.float32) / 255.0
        
        gender, spectacles, facial_hair, pose, emotion = self.__extract_csv(index)
        label = image_path.parent.parent.name
        label_index = self.classes.index(label)
        
        if self.transform:
            numpy_image = self.transform(image = numpy_image)['image']
        
        # Đổi tử HWC sang CHW cho torch
        X = torch.from_numpy(numpy_image).permute(2,0,1)
        y = torch.tensor([label_index, gender, spectacles, facial_hair, pose, emotion], dtype=torch.int)
        return X, y
        
    def __load_numpy_image(self, image_path:str):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {image_path}")
            
        # Xử lý kênh màu theo từng type
        if self.type in ['albedo', 'depthmap']:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Chuyển grayscale sang 3 kênh (RGB) nếu mô hình yêu cầu đầu vào 3 kênh
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    def __extract_csv(self, index: int):
        row = self.df.iloc[index]
        return row["Gender"], row["Spectacles"], row["Facial_Hair"], row["Pose"], row["Emotion"]

    def __calculate_weight_class(self):
        df = pd.read_csv(self.metadata_path)
        
        label_columns = ['id', 'Gender', 'Spectacles', 'Facial_Hair', 'Pose', 'Emotion']
        
         # Calculate the relative frequency and alpha_t for each class in each label column
        weight_class = {}
        for column in label_columns:
            value_counts = df[column].value_counts(normalize=True)  # Tần suất tương đối
            alpha_t = (1 / value_counts) / (1 / value_counts).sum()  # Normalize để tổng bằng 1
            weight_class[column] = alpha_t.to_dict()
        
        return weight_class


class ConcatCustomPhotometricDatasetV3(Dataset):
    
    def __init__(self, dataset_dir:str, transform, train = True):
        self.dataset_dir = Path(dataset_dir)
        
        split = 'train' if train else 'val'
        
        self.metadata_path = self.dataset_dir / f"{split}_set.csv"
                
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metadata tại: {self.metadata_path}")
        
        # Đọc file CSV MỘT LẦN DUY NHẤT và lưu vào self.df
        self.df = pd.read_csv(self.metadata_path)
        
        self.transform = transform
        
        # 2. Ánh xạ type sang tên file ảnh mới
        filename_mapping = {
            'albedo': 'albedo_crop.png',
            'normalmap': 'normalmap_crop.png',
            'depthmap': 'depthmap_crop.png',
        }
        
        target_filename_1 = filename_mapping['albedo']
        target_filename_2 = filename_mapping['normalmap']
        target_filename_3 = filename_mapping['depthmap']
        
        # 3. Lấy danh sách classes từ cột 'id' trong CSV
        self.classes = sorted(self.df['id'].astype(str).unique().tolist())
        self.class_to_idx = {
            class_id: i
            for i, class_id in enumerate(self.classes)
        }
        
        # 4. Tạo danh sách self.paths từ file CSV
        self.data = []
        
        for _, row in self.df.iterrows():
            subject_id = str(row['id'])
            session = str(row['session'])
            img_path_1 = self.dataset_dir / subject_id / session / target_filename_1
            img_path_2 = self.dataset_dir / subject_id / session / target_filename_2
            img_path_3 = self.dataset_dir / subject_id / session / target_filename_3
            
            class_index = self.class_to_idx[subject_id]
            
            self.data.append((img_path_1, img_path_2, img_path_3, class_index))
    
        # 5. Kiểm tra file
        missing_files = [
            str(p)
            for item in self.data
            for p in item[:3]
            if not p.is_file()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Thiếu {len(missing_files)} file ảnh. "
                f"Ví dụ: {missing_files[:6]}"
            )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        albedo_path, normal_path, depth_path, label_index = self.data[index]
        numpy_normal = self.__load_numpy_image(normal_path)
        numpy_albedo = self.__load_numpy_image(albedo_path)
        numpy_depth = self.__load_numpy_image(depth_path)
        
        # === 1. CHUẨN HÓA VỀ [0, 1] VÀ ĐỔI SANG FLOAT32 ===
        # Vì ảnh đang ở uint8 (0-255), chia 255.0 để đưa về [0.0, 1.0]
        numpy_normal = numpy_normal.astype(np.float32) / 255.0
        numpy_albedo = numpy_albedo.astype(np.float32) / 255.0
        numpy_depth = numpy_depth.astype(np.float32) / 255.0
        
        row = self.df.iloc[index]
        
        gender = int(row['Gender'])
        spectacles = int(row['Spectacles'])
        facial_hair = int(row['Facial_Hair'])
        pose = int(row['Pose'])
        emotion = int(row['Emotion'])
        
        if self.transform:
            transformed = self.transform(image=numpy_normal, albedo=numpy_albedo, depthmap=numpy_depth)
            numpy_normal = transformed['image']
            numpy_albedo = transformed['albedo']
            numpy_depth = transformed['depthmap']
        
        # Stack các tensor lại thành một tensor duy nhất
        X = torch.stack((
            torch.from_numpy(numpy_normal).permute(2, 0, 1),
            torch.from_numpy(numpy_albedo).permute(2, 0, 1), 
            torch.from_numpy(numpy_depth).permute(2, 0, 1)
        ), dim=0)
        y = torch.tensor([label_index, gender, spectacles, facial_hair, pose, emotion], dtype=torch.int)
        return X, y, index
        
    def __load_numpy_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                f"Unsupported image shape: {image.shape}, "
                f"path={image_path}"
            )
        
        return image


class ConcatCustomPhotometricDatasetV2(Dataset):
    
    def __init__(self, dataset_dir:str, type_1:str, type_2:str, transform, train = True):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.type_1 = type_1.lower()
        self.type_2 = type_2.lower()
        
        # 1. Xác định file CSV dựa vào tham số train
        split = 'train' if train else 'val' # Hoặc 'test' tùy ý bạn
        self.metadata_path = self.dataset_dir / f"{split}_set.csv"
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metadata tại: {self.metadata_path}")
            
        # Đọc file CSV MỘT LẦN DUY NHẤT và lưu vào self.df
        self.df = pd.read_csv(self.metadata_path)
        
        # 2. Ánh xạ type sang tên file ảnh mới
        filename_mapping = {
            'normalmap': 'normalmap_crop.png',
            'depthmap': 'depthmap_crop.png',
            'albedo': 'albedo_crop.png'
        }
        
        if self.type_1 not in filename_mapping:
            raise ValueError(f"Sai type_1: {type_1}")

        if self.type_2 not in filename_mapping:
            raise ValueError(f"Sai type_2: {type_2}")
            
        target_filename_1 = filename_mapping[self.type_1]
        target_filename_2 = filename_mapping[self.type_2]
        
        # 3. Lấy danh sách classes từ cột 'id' trong CSV
        self.classes = sorted(self.df['id'].astype(str).unique().tolist())
        self.class_to_idx = {
            class_id: i
            for i, class_id in enumerate(self.classes)
        }
        
        # 4. Tạo danh sách self.paths từ file CSV
        self.data = []
        
        for _, row in self.df.iterrows():
            subject_id = str(row['id'])
            session = str(row['session'])
            img_path_1 = self.dataset_dir / subject_id / session / target_filename_1
            img_path_2 = self.dataset_dir / subject_id / session / target_filename_2
            
            class_index = self.class_to_idx[subject_id]
            
            self.data.append((img_path_1, img_path_2, class_index))
   
        # 5. Kiểm tra file
        missing_files = [
            str(p)
            for item in self.data
            for p in item[:2]
            if not p.is_file()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Thiếu {len(missing_files)} file ảnh. "
                f"Ví dụ: {missing_files[:5]}"
            )
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        type1_path, type2_path, label_index = self.data[index]
        numpy_type1 = self.__load_numpy_image(type1_path)
        numpy_type2 = self.__load_numpy_image(type2_path)
        
        # === 1. CHUẨN HÓA VỀ [0, 1] VÀ ĐỔI SANG FLOAT32 ===
        # Vì ảnh đang ở uint8 (0-255), chia 255.0 để đưa về [0.0, 1.0]
        numpy_type1 = numpy_type1.astype(np.float32) / 255.0
        numpy_type2 = numpy_type2.astype(np.float32) / 255.0
        
        row = self.df.iloc[index]

        gender = int(row['Gender'])
        spectacles = int(row['Spectacles'])
        facial_hair = int(row['Facial_Hair'])
        pose = int(row['Pose'])
        emotion = int(row['Emotion'])
        
        if self.transform:
            transformed = self.transform(image=numpy_type1, image2=numpy_type2)
            numpy_type1 = transformed['image']
            numpy_type2 = transformed['image2']
        
        # Stack các tensor lại thành một tensor duy nhất
        X = torch.stack((
            torch.from_numpy(numpy_type1).permute(2, 0, 1),
            torch.from_numpy(numpy_type2).permute(2, 0, 1), 
        ), dim=0)
        y = torch.tensor([label_index, gender, spectacles, facial_hair, pose, emotion], dtype=torch.int)
        return X, y
        
    def __load_numpy_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                f"Unsupported image shape: {image.shape}, "
                f"path={image_path}"
            )
        
        return image


class UniqueIdBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.unique_labels = list(set(labels))
        self.label_indices = {l: [] for l in self.unique_labels}
        for idx, l in enumerate(self.labels): self.label_indices[l].append(idx)
    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            batch_labels = np.random.choice(self.unique_labels, size=self.batch_size, replace=False)
            batch_indices = [np.random.choice(self.label_indices[l]) for l in batch_labels]
            yield batch_indices
    def __len__(self): return len(self.labels) // self.batch_size


def create_multitask_dataloader(config, train_transform, test_transform) -> Tuple[DataLoader, DataLoader]:
    
    train_data = CustomPhotometricDataset(config['dataset_dir'], train_transform, config['type'])
    test_data = CustomPhotometricDataset(config['dataset_dir'], test_transform, config['type'], train=False)

    use_sampler = config.get('use_sampler', False)
    
    if use_sampler:
        print(f">>> SingleLoader: MODE = PK SAMPLER")
        df = pd.read_csv(os.path.join(config['dataset_dir'], 'train_set.csv'))
        unique_ids = sorted(df['id'].unique())
        id_to_label = {id_val: i for i, id_val in enumerate(unique_ids)}
        labels_list = [id_to_label[row['id']] for _, row in df.iterrows()]
        train_dataloader = DataLoader(
            train_data,
            batch_sampler=UniqueIdBatchSampler(labels_list, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
        )
    else:    
        train_dataloader = DataLoader(
            train_data,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
        )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=config['num_workers'],
    )
    
    return train_dataloader, test_dataloader, train_data.weightclass


def create_concatv2_multitask_dataloader(config, train_transform, test_transform) ->Tuple[DataLoader, DataLoader]:
    train_data = ConcatCustomPhotometricDatasetV2(config['dataset_dir'], config['type_1'], config['type_2'], train_transform)
    test_data = ConcatCustomPhotometricDatasetV2(config['dataset_dir'], config['type_1'], config['type_2'], test_transform, train=False)
    
    use_sampler = config.get('use_sampler', False)
        
    if use_sampler:
        print(f">>> SingleLoader: MODE = PK SAMPLER")
        df = pd.read_csv(os.path.join(config['dataset_dir'], 'train_set.csv'))
        unique_ids = sorted(df['id'].unique())
        id_to_label = {id_val: i for i, id_val in enumerate(unique_ids)}
        labels_list = [id_to_label[row['id']] for _, row in df.iterrows()]
        train_dataloader = DataLoader(
            train_data,
            batch_sampler=UniqueIdBatchSampler(labels_list, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
        )
    else:    
        train_dataloader = DataLoader(
            train_data,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
        )
        
    test_dataloader = DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    return train_dataloader, test_dataloader


def create_concatv3_multitask_dataloader(config, train_transform, test_transform) ->Tuple[DataLoader, DataLoader]:
    train_data = ConcatCustomPhotometricDatasetV3(config['dataset_dir'], train_transform)
    test_data = ConcatCustomPhotometricDatasetV3(config['dataset_dir'], test_transform, train=False)
    
    use_sampler = config.get('use_sampler', False)
        
    if use_sampler:
        print(f">>> SingleLoader: MODE = PK SAMPLER")
        df = pd.read_csv(os.path.join(config['dataset_dir'], 'train_set.csv'))
        unique_ids = sorted(df['id'].unique())
        id_to_label = {id_val: i for i, id_val in enumerate(unique_ids)}
        labels_list = [id_to_label[row['id']] for _, row in df.iterrows()]
        train_dataloader = DataLoader(
            train_data,
            batch_sampler=UniqueIdBatchSampler(labels_list, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
        )
    else:    
        train_dataloader = DataLoader(
            train_data,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
        )
            
    test_dataloader = DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )
    return train_dataloader, test_dataloader