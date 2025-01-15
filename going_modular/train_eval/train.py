import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ..utils.roc_auc import compute_auc
from ..utils.metrics import ProgressMeter
from ..utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ..utils.ModelCheckPoint import ModelCheckpoint
import os

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)


def fit(
    conf: dict,
    start_epoch: int,
    model: Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    scheduler,
    early_stopping: MultiMetricEarlyStopping,
    model_checkpoint: ModelCheckpoint
):
    log_dir = os.path.abspath(conf['checkpoint_dir'] + conf['type'] + '/logs')
    writer = SummaryWriter(log_dir=log_dir)
    device = conf['device']
    
    for epoch in range(start_epoch, conf['epochs']):
        
        (   
            train_loss,
            train_loss_id,
            train_loss_gender,
            train_loss_da_gender,
            train_loss_emotion,
            train_loss_da_emotion,
            train_loss_pose,
            train_loss_da_pose,
            train_loss_facial_hair,
            train_loss_da_facial_hair,
            train_loss_spectacles,
            train_loss_da_spectacles
        ) = train_epoch(train_dataloader, model, criterion, optimizer, device)
        
        (   
            test_loss_gender,
            test_loss_da_gender,
            test_loss_emotion,
            test_loss_da_emotion,
            test_loss_pose,
            test_loss_da_pose,
            test_loss_facial_hair,
            test_loss_da_facial_hair,
            test_loss_spectacles,
            test_loss_da_spectacles
        ) = test_epoch(test_dataloader, model, criterion, device)
        
        train_auc = compute_auc(train_dataloader, model, device)
        train_gender_auc = train_auc['gender']
        train_spectacles_auc = train_auc['spectacles']
        train_facial_hair_auc = train_auc['facial_hair']
        train_pose_auc = train_auc['pose']
        train_emotion_auc = train_auc['emotion']
        train_id_cosine_auc = train_auc['id_cosine']
        train_id_euclidean_auc = train_auc['id_euclidean']
        
        test_auc = compute_auc(test_dataloader, model, device)
        test_gender_auc = test_auc['gender']
        test_spectacles_auc = test_auc['spectacles']
        test_facial_hair_auc = test_auc['facial_hair']
        test_pose_auc = test_auc['pose']
        test_emotion_auc = test_auc['emotion']
        test_id_cosine_auc = test_auc['id_cosine']
        test_id_euclidean_auc = test_auc['id_euclidean']
        
        # Log các giá trị vào TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        
        writer.add_scalars(main_tag='Loss/gender', tag_scalar_dict={
            'train': train_loss_gender, 'test': test_loss_gender}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/da_gender', tag_scalar_dict={
            'train': train_loss_da_gender, 'test': test_loss_da_gender}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/spectacles', tag_scalar_dict={
            'train': train_loss_spectacles, 'test': test_loss_spectacles}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/da_spectacles', tag_scalar_dict={
            'train': train_loss_da_spectacles, 'test': test_loss_da_spectacles}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/facial_hair', tag_scalar_dict={
            'train': train_loss_facial_hair, 'test': test_loss_facial_hair}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/da_facial_hair', tag_scalar_dict={
            'train': train_loss_da_facial_hair, 'test': test_loss_da_facial_hair}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/pose', tag_scalar_dict={
            'train': train_loss_pose, 'test': test_loss_pose}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/da_pose', tag_scalar_dict={
            'train': train_loss_da_pose, 'test': test_loss_da_pose}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/emotion', tag_scalar_dict={
            'train': train_loss_emotion, 'test': test_loss_emotion}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='Loss/da_emotion', tag_scalar_dict={
            'train': train_loss_da_emotion, 'test': test_loss_da_emotion}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='AUC/cosine', tag_scalar_dict={
            'train': train_id_cosine_auc, 'test': test_id_cosine_auc}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='AUC/euclidean', tag_scalar_dict={
            'train': train_id_euclidean_auc, 'test': test_id_euclidean_auc}, global_step=epoch+1)

        writer.add_scalars(main_tag='AUC/gender', tag_scalar_dict={
            'train': train_gender_auc, 'test': test_gender_auc}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='AUC/spectacles', tag_scalar_dict={
            'train': train_spectacles_auc, 'test': test_spectacles_auc}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='AUC/facial_hair', tag_scalar_dict={
            'train': train_facial_hair_auc, 'test': test_facial_hair_auc}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='AUC/pose', tag_scalar_dict={
            'train': train_pose_auc, 'test': test_pose_auc}, global_step=epoch+1)
        
        writer.add_scalars(main_tag='AUC/emotion', tag_scalar_dict={
            'train': train_emotion_auc, 'test': test_emotion_auc}, global_step=epoch+1)
        
        train_metrics = {
            "loss": train_loss,
            "loss_id": train_loss_id,
            "loss_gender": train_loss_gender,
            "loss_da_gender": train_loss_da_gender,
            "loss_emotion": train_loss_emotion,
            "loss_da_emotion": train_loss_da_emotion,
            "loss_pose": train_loss_pose,
            "loss_da_pose": train_loss_da_pose,
            "loss_facial_hair": train_loss_facial_hair,
            "loss_da_facial_hair": train_loss_da_facial_hair,
            "loss_spectacles": train_loss_spectacles,
            "loss_da_spectacles": train_loss_da_spectacles,
            "auc_gender": train_gender_auc,
            "auc_spectacles": train_spectacles_auc,
            "auc_facial_hair": train_facial_hair_auc,
            "auc_pose": train_pose_auc,
            "auc_emotion": train_emotion_auc,
            "auc_id_cosine": train_id_cosine_auc,
            "auc_id_euclidean": train_id_euclidean_auc,
        }

        test_metrics = {
            "loss_gender": test_loss_gender,
            "loss_da_gender": test_loss_da_gender,
            "loss_emotion": test_loss_emotion,
            "loss_da_emotion": test_loss_da_emotion,
            "loss_pose": test_loss_pose,
            "loss_da_pose": test_loss_da_pose,
            "loss_facial_hair": test_loss_facial_hair,
            "loss_da_facial_hair": test_loss_da_facial_hair,
            "loss_spectacles": test_loss_spectacles,
            "loss_da_spectacles": test_loss_da_spectacles,
            "auc_gender": test_gender_auc,
            "auc_spectacles": test_spectacles_auc,
            "auc_facial_hair": test_facial_hair_auc,
            "auc_pose": test_pose_auc,
            "auc_emotion": test_emotion_auc,
            "auc_id_cosine": test_id_cosine_auc,
            "auc_id_euclidean": test_id_euclidean_auc,
        }


        process = ProgressMeter(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            prefix=f"Epoch {epoch + 1}:"
        )

        process.display()

        model_checkpoint(model, optimizer, epoch + 1)
        early_stopping([test_id_cosine_auc, test_id_euclidean_auc], model, epoch + 1)
        
        # if early_max_stopping.early_stop and early_min_stopping.early_stop:
        #     break
        
        scheduler.step(epoch)
        
    writer.close()
 
    
def train_epoch(
    train_dataloader: DataLoader, 
    model: Module, 
    criterion: Module, 
    optimizer: Optimizer, 
    device: str,
):
    model.to(device)
    model.train()

    train_loss = 0

    # Các biến lưu trữ tổng loss từng loại
    total_loss_id = 0
    total_loss_gender = 0
    total_loss_da_gender = 0
    total_loss_emotion = 0
    total_loss_da_emotion = 0
    total_loss_pose = 0
    total_loss_da_pose = 0
    total_loss_facial_hair = 0
    total_loss_da_facial_hair = 0
    total_loss_spectacles = 0
    total_loss_da_spectacles = 0

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        (
            total_loss, 
            loss_id, loss_gender, loss_da_gender, 
            loss_emotion, loss_da_emotion, 
            loss_pose, loss_da_pose, 
            loss_facial_hair, loss_da_facial_hair, 
            loss_spectacles, loss_da_spectacles
        ) = criterion(logits, y)

        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

        # Cộng dồn các giá trị loss riêng biệt
        total_loss_id += loss_id.item()
        total_loss_gender += loss_gender.item()
        total_loss_da_gender += loss_da_gender.item()
        total_loss_emotion += loss_emotion.item()
        total_loss_da_emotion += loss_da_emotion.item()
        total_loss_pose += loss_pose.item()
        total_loss_da_pose += loss_da_pose.item()
        total_loss_facial_hair += loss_facial_hair.item()
        total_loss_da_facial_hair += loss_da_facial_hair.item()
        total_loss_spectacles += loss_spectacles.item()
        total_loss_da_spectacles += loss_da_spectacles.item()

    # Tính trung bình của từng loại loss
    num_batches = len(train_dataloader)
    avg_train_loss = train_loss / num_batches
    avg_loss_id = total_loss_id / num_batches
    avg_loss_gender = total_loss_gender / num_batches
    avg_loss_da_gender = total_loss_da_gender / num_batches
    avg_loss_emotion = total_loss_emotion / num_batches
    avg_loss_da_emotion = total_loss_da_emotion / num_batches
    avg_loss_pose = total_loss_pose / num_batches
    avg_loss_da_pose = total_loss_da_pose / num_batches
    avg_loss_facial_hair = total_loss_facial_hair / num_batches
    avg_loss_da_facial_hair = total_loss_da_facial_hair / num_batches
    avg_loss_spectacles = total_loss_spectacles / num_batches
    avg_loss_da_spectacles = total_loss_da_spectacles / num_batches
    
    return (
        avg_train_loss,
        avg_loss_id,
        avg_loss_gender,
        avg_loss_da_gender,
        avg_loss_emotion,
        avg_loss_da_emotion,
        avg_loss_pose,
        avg_loss_da_pose,
        avg_loss_facial_hair,
        avg_loss_da_facial_hair,
        avg_loss_spectacles,
        avg_loss_da_spectacles,
    )


def test_epoch(
    test_dataloader: DataLoader, 
    model: Module, 
    criterion: Module, 
    device: str,
):
    model.to(device)
    model.eval()

    # Các biến lưu trữ tổng loss từng loại
    total_loss_gender = 0
    total_loss_da_gender = 0
    total_loss_emotion = 0
    total_loss_da_emotion = 0
    total_loss_pose = 0
    total_loss_da_pose = 0
    total_loss_facial_hair = 0
    total_loss_da_facial_hair = 0
    total_loss_spectacles = 0
    total_loss_da_spectacles = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            (
                total_loss, 
                loss_id, loss_gender, loss_da_gender, 
                loss_emotion, loss_da_emotion, 
                loss_pose, loss_da_pose, 
                loss_facial_hair, loss_da_facial_hair, 
                loss_spectacles, loss_da_spectacles
            ) = criterion(logits, y)


            # Cộng dồn các giá trị loss riêng biệt
            total_loss_gender += loss_gender.item()
            total_loss_da_gender += loss_da_gender.item()
            total_loss_emotion += loss_emotion.item()
            total_loss_da_emotion += loss_da_emotion.item()
            total_loss_pose += loss_pose.item()
            total_loss_da_pose += loss_da_pose.item()
            total_loss_facial_hair += loss_facial_hair.item()
            total_loss_da_facial_hair += loss_da_facial_hair.item()
            total_loss_spectacles += loss_spectacles.item()
            total_loss_da_spectacles += loss_da_spectacles.item()

    # Tính trung bình của từng loại loss
    num_batches = len(test_dataloader)
    avg_loss_gender = total_loss_gender / num_batches
    avg_loss_da_gender = total_loss_da_gender / num_batches
    avg_loss_emotion = total_loss_emotion / num_batches
    avg_loss_da_emotion = total_loss_da_emotion / num_batches
    avg_loss_pose = total_loss_pose / num_batches
    avg_loss_da_pose = total_loss_da_pose / num_batches
    avg_loss_facial_hair = total_loss_facial_hair / num_batches
    avg_loss_da_facial_hair = total_loss_da_facial_hair / num_batches
    avg_loss_spectacles = total_loss_spectacles / num_batches
    avg_loss_da_spectacles = total_loss_da_spectacles / num_batches

    # Trả về tất cả các loss dưới dạng tuple
    return (
        avg_loss_gender,
        avg_loss_da_gender,
        avg_loss_emotion,
        avg_loss_da_emotion,
        avg_loss_pose,
        avg_loss_da_pose,
        avg_loss_facial_hair,
        avg_loss_da_facial_hair,
        avg_loss_spectacles,
        avg_loss_da_spectacles,
    )
