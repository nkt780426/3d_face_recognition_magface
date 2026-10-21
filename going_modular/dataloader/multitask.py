import torch
from torch.utils.data import Dataset, DataLoader

import cv2, os
from pathlib import Path
from typing import Tuple
import random
import pandas as pd

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)
random.seed(seed)

class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)  # Tạo iterator cho DataLoader
        self.stream = torch.cuda.Stream()  # Tạo stream CUDA để tải trước dữ liệu
        self.preload()  # Tải trước dữ liệu đầu tiên

    def preload(self):
        try:
            self.next_data = next(self.loader)  # Lấy batch tiếp theo từ DataLoader
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            # Tải dữ liệu vào GPU
            self.next_input = self.next_data[0].cuda(non_blocking=True)
            self.next_target = self.next_data[1].cuda(non_blocking=True)

    def next(self):
        # Đồng bộ hóa stream hiện tại với stream prefetch
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()  # Tải trước batch tiếp theo
        return input, target


class CustomExrDataset(Dataset):
    
    def __init__(self, dataset_dir:str, transform, type='normalmap', train = True):
        '''
            type = ['normalmap', 'depthmap', 'albedo']
        '''
        self.metadata_path = os.path.join(dataset_dir, 'train_set.csv') if train else os.path.join(dataset_dir, 'test_set.csv')
        split = 'train' if train else 'test'
        if type == 'normalmap':
            dataset_dir = Path(dataset_dir, 'Normal_Map', split)
        elif type == 'depthmap':
            dataset_dir = Path(dataset_dir, 'Depth_Map', split)
        elif type == 'albedo':
            dataset_dir = Path(dataset_dir, 'Albedo', split)
        else:
            raise Exception(f'Sai trường type: {type}')
        
        self.paths = list(Path(dataset_dir).glob("*/*.exr"))
        self.transform = transform
        self.type = type
        self.classes = sorted(os.listdir(dataset_dir))
        
        self.weightclass = self.__calculate_weight_class()
        
    def __len__(self):
        return len(self.paths)
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        image_path = self.paths[index]
        numpy_image = self.__load_numpy_image(image_path)
        gender, spectacles, facial_hair, pose, emotion = self.__extract_csv(image_path)
        label = image_path.parent.name
        label_index = self.classes.index(label)
        
        if self.transform:
            numpy_image = self.transform(image = numpy_image)['image']
        
        X = torch.from_numpy(numpy_image).permute(2,0,1)
        y = torch.tensor([label_index, gender, spectacles, facial_hair, pose, emotion], dtype=torch.int)
        return X, y
        
    def __load_numpy_image(self, image_path:str):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        if self.type in ['albedo', 'depthmap']:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    def __extract_csv(self, image_path):
        id = image_path.parent.name
        session = image_path.stem
        df = pd.read_csv(self.metadata_path)
        
        # Lọc dữ liệu theo ID và session
        filtered_data = df[(df['id'] == int(id)) & (df['session'] == session)]
        
        # Kiểm tra nếu không có hoặc có nhiều hơn 1 dòng được trả về
        if filtered_data.shape[0] != 1:
            raise Exception(f"Tìm thấy {filtered_data.shape[0]} row có {id} và {session} trong file {self.metadata_path}")
        
        row = filtered_data.iloc[0]  # Lấy dòng đầu tiên (vì chỉ có 1 dòng được trả về)
        
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


class ConcatCustomExrDatasetV3(Dataset):
    
    def __init__(self, dataset_dir:str, transform, train = True):
        self.metadata_path = os.path.join(dataset_dir, 'train_set.csv') if train else os.path.join(dataset_dir, 'test_set.csv')
        split = 'train' if train else 'test'
        self.albedo_dir = Path(dataset_dir) / 'Albedo' / split
        self.depth_dir = Path(dataset_dir) / 'Depth_Map' / split
        self.normal_dir = Path(dataset_dir) / 'Normal_Map' / split
        
        self.transform = transform
        self.classes = sorted(os.listdir(self.albedo_dir))
        
        # Collect paths for each modality
        self.data = []
        for class_name in self.classes:
            albedo_class_dir = self.albedo_dir / class_name
            depth_class_dir = self.depth_dir / class_name
            normal_class_dir = self.normal_dir / class_name

            albedo_files = sorted(list(albedo_class_dir.glob("*.exr")))
            depth_files = sorted(list(depth_class_dir.glob("*.exr")))
            normal_files = sorted(list(normal_class_dir.glob("*.exr")))

            assert len(albedo_files) == len(normal_files) == len(depth_files), (
                f"Mismatch in number of files for class {class_name}: Albedo({len(albedo_files)}), "
                f"Normal({len(normal_files)}), Depth({len(depth_files)})"
            )
            class_index = self.classes.index(class_name)
            for albedo_path, normal_path, depth_path in zip(albedo_files, normal_files, depth_files):
                self.data.append((albedo_path, normal_path, depth_path, class_index))
   
                
    def __len__(self):
        return len(self.data)
    
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        albedo_path, normal_path, depth_path, label_index = self.data[index]
        numpy_normal = self.__load_numpy_image(normal_path)
        numpy_albedo = self.__load_numpy_image(albedo_path)
        numpy_depth = self.__load_numpy_image(depth_path)
        gender, spectacles, facial_hair, pose, emotion = self.__extract_csv(albedo_path)
        
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
        return X, y
        
        
    def __load_numpy_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image


    def __extract_csv(self, image_path):
        id = image_path.parent.name
        session = image_path.stem
        df = pd.read_csv(self.metadata_path)
        
        # Lọc dữ liệu theo ID và session
        filtered_data = df[(df['id'] == int(id)) & (df['session'] == session)]
        
        # Kiểm tra nếu không có hoặc có nhiều hơn 1 dòng được trả về
        if filtered_data.shape[0] != 1:
            raise Exception(f"Tìm thấy {filtered_data.shape[0]} row có {id} và {session} trong file {self.metadata_path}")
        
        row = filtered_data.iloc[0]  # Lấy dòng đầu tiên (vì chỉ có 1 dòng được trả về)
        
        return row["Gender"], row["Spectacles"], row["Facial_Hair"], row["Pose"], row["Emotion"]


class ConcatCustomExrDatasetV2(Dataset):
    
    
    def __init__(self, dataset_dir_1:str, dataset_dir_2:str, transform, train = True):
        dataset_dir = os.path.dirname(dataset_dir_1)
        self.metadata_path = os.path.join(dataset_dir, 'train_set.csv') if train else os.path.join(dataset_dir, 'test_set.csv')
        split = 'train' if train else 'test'
        
        self.dataset_dir1 = Path(dataset_dir_1) / split
        self.dataset_dir2 = Path(dataset_dir_2) / split
        
        self.transform = transform
        self.classes = sorted(os.listdir(self.dataset_dir1))
        
        # Collect paths for each modality
        self.data = []
        for class_name in self.classes:
            class_dir1 = self.dataset_dir1 / class_name
            class_dir2 = self.dataset_dir2 / class_name

            dir1_files = sorted(list(class_dir1.glob("*.exr")))
            dir2_files = sorted(list(class_dir2.glob("*.exr")))

            assert len(dir1_files) == len(dir2_files), (
                f"Mismatch in number of files for class {class_name}: dataset 1 ({len(dir1_files)}), "
                f"dataset 2({len(dir2_files)})"
            )
            class_index = self.classes.index(class_name)
            for dir1_path, dir2_path in zip(dir1_files, dir2_files):
                self.data.append((dir1_path, dir2_path, class_index))
   
                
    def __len__(self):
        return len(self.data)
    
    
    # Nhận vào index mà dataloader muốn lấy
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        type1_path, type2_path, label_index = self.data[index]
        numpy_type1 = self.__load_numpy_image(type1_path)
        numpy_type2 = self.__load_numpy_image(type2_path)
        gender, spectacles, facial_hair, pose, emotion = self.__extract_csv(type1_path)
        
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
        
        return image


    def __extract_csv(self, image_path):
        id = image_path.parent.name
        session = image_path.stem
        df = pd.read_csv(self.metadata_path)
        
        # Lọc dữ liệu theo ID và session
        filtered_data = df[(df['id'] == int(id)) & (df['session'] == session)]
        
        # Kiểm tra nếu không có hoặc có nhiều hơn 1 dòng được trả về
        if filtered_data.shape[0] != 1:
            raise Exception(f"Tìm thấy {filtered_data.shape[0]} row có {id} và {session} trong file {self.metadata_path}")
        
        row = filtered_data.iloc[0]  # Lấy dòng đầu tiên (vì chỉ có 1 dòng được trả về)
        
        return row["Gender"], row["Spectacles"], row["Facial_Hair"], row["Pose"], row["Emotion"]
    
    
def create_multitask_datafetcher(config, train_transform, test_transform) -> Tuple[DataLoader, DataLoader]:
    
    train_data = CustomExrDataset(config['dataset_dir'], train_transform, config['type'])
    test_data = CustomExrDataset(config['dataset_dir'], test_transform, config['type'], train=False)

    train_dataloader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    # train_datafetcher = DataPrefetcher(train_dataloader)
    # test_datafetcher = DataPrefetcher(test_dataloader)
    
    
    return train_dataloader, test_dataloader, train_data.weightclass


def create_concatv2_multitask_datafetcher(config, train_transform, test_transform) ->Tuple[DataLoader, DataLoader]:
    train_data = ConcatCustomExrDatasetV2(config['dataset_dir_1'], config['dataset_dir_2'], train_transform)
    test_data = ConcatCustomExrDatasetV2(config['dataset_dir_1'], config['dataset_dir_2'], test_transform, train=False)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def create_concatv3_multitask_datafetcher(config, train_transform, test_transform) ->Tuple[DataLoader, DataLoader]:
    train_data = ConcatCustomExrDatasetV3(config['dataset_dir'], train_transform)
    test_data = ConcatCustomExrDatasetV3(config['dataset_dir'], test_transform, train=False)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return train_dataloader, test_dataloader