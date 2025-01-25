import torch
import numpy as np
from sklearn.metrics import roc_auc_score

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

def compute_auc(
    dataloader: torch.utils.data.DataLoader, 
    model: torch.nn.Module, 
    device: str
):
    
    model.eval()
    all_labels = {'gender': [], 'spectacles': [], 'facial_hair': [], 'pose': [], 'emotion': [] }
    all_preds = {'gender': [], 'spectacles': [], 'facial_hair': [], 'pose': [], 'emotion': [] }

    with torch.no_grad():
        embeddings_list = []

        for batch in dataloader:
            images, y = batch
            # Lấy các nhãn thực tế từ y
            id, gender, spectacles, facial_hair, pose, emotion = (
                y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5]
            )
            
            images = images.to(device)
            # Trừ id, còn lại đều đã qua softmax
            x_id, x_gender, x_pose, x_emotion, x_facial_hair, x_spectacles = model.get_result(images)

            # Append IDs and embeddings
            embeddings_list.append((id, x_id))

            # Tính xác suất và lớp dự đoán cho mỗi thuộc tính (gender, spectacles, facial_hair, pose, occlusion, emotion)
            for attribute, x_attr, y_attr in zip(
                ['gender', 'spectacles', 'facial_hair', 'pose', 'emotion'],
                [torch.softmax(x_gender, dim=1), torch.softmax(x_spectacles, dim=1), torch.softmax(x_facial_hair, dim=1), 
                torch.softmax(x_pose, dim=1), torch.softmax(x_emotion, dim=1)],
                [gender, spectacles, facial_hair, pose, emotion]
            ):
                # Tìm xác suất lớn nhất và lớp của nó
                probs, predicted_classes = torch.max(x_attr, dim=1)

                # Đảm bảo y_attr cùng device với predicted_classes
                y_attr = y_attr.to(predicted_classes.device)

                # Cập nhật all_labels với việc so sánh lớp dự đoán với nhãn thực tế
                all_labels[attribute].append((predicted_classes == y_attr).int().cpu().numpy())

                # Lấy xác suất cho lớp dự đoán đúng từ x_attr
                all_preds[attribute].append(torch.gather(x_attr, 1, y_attr.unsqueeze(1).long()).squeeze(1).cpu().numpy())

        # Chuyển thành mảng 1 chiều cho tất cả các lớp
        for attribute in all_labels:
            all_labels[attribute] = np.concatenate(all_labels[attribute], axis=0)
        for attribute in all_preds:
            all_preds[attribute] = np.concatenate(all_preds[attribute], axis=0)


        # Concatenate all id embeddings into one tensor
        all_ids = torch.cat([x[0] for x in embeddings_list], dim=0)
        all_embeddings = torch.cat([x[1] for x in embeddings_list], dim=0)
        
        euclidean_scores = []
        euclidean_labels = []
        cosine_scores = []
        cosine_labels = []

        # Compute pairwise Euclidean distance and cosine similarity
        all_embeddings_norm = all_embeddings / all_embeddings.norm(p=2, dim=1, keepdim=True)
        euclidean_distances = torch.cdist(all_embeddings, all_embeddings, p=2)  # Euclidean distance matrix
        cosine_similarities = torch.mm(all_embeddings_norm, all_embeddings_norm.t())  # Cosine similarity matrix
        
        # Compute labels (same id = 0, different id = 1)
        labels = (all_ids.unsqueeze(1) == all_ids.unsqueeze(0)).int().to(device)

        # Flatten and filter results
        euclidean_scores = euclidean_distances[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        euclidean_labels = labels[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        
        cosine_scores = cosine_similarities[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        cosine_labels = labels[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        
        # Compute ROC AUC for Euclidean distance
        all_labels['id_euclidean'] = 1 - np.array(euclidean_labels)
        all_preds['id_euclidean'] = np.array(euclidean_scores)

        # Compute ROC AUC for Cosine similarity
        all_labels['id_cosine'] = np.array(cosine_labels)
        all_preds['id_cosine'] = np.array(cosine_scores)
        
        # # Calculate accuracy for Euclidean distance
        # euclidean_optimal_idx = np.argmax(tpr_euclidean - fpr_euclidean) # Chọn ngưỡng tại điểm có giá trị tpr - fpr lớn nhất trên đường ROC, vì đây là nơi tối ưu hóa sự cân bằng giữa tỷ lệ phát hiện (TPR) và tỷ lệ báo động giả (FPR).
        # euclidean_optimal_threshold = thresholds_euclidean[euclidean_optimal_idx]
        # euclidean_pred_labels = (euclidean_pred_scores >= euclidean_optimal_threshold).astype(int)
        # euclidean_accuracy = accuracy_score(euclidean_true_labels, euclidean_pred_labels)

        # # Calculate accuracy for Cosine similarity
        # cosine_optimal_idx = np.argmax(tpr_cosine - fpr_cosine)
        # cosine_optimal_threshold = thresholds_cosine[cosine_optimal_idx]
        # cosine_pred_labels = (cosine_pred_scores >= cosine_optimal_threshold).astype(int)
        # cosine_accuracy = accuracy_score(cosine_true_labels, cosine_pred_labels)
        
        # Tính AUC cho từng tác vụ
        auc_scores = {}
        for task in all_labels:
            unique_classes = np.unique(all_labels[task])
            if len(unique_classes) < 2:
                # Nếu chỉ có một lớp, gán AUC là None hoặc giá trị mặc định
                auc_scores[task] = 1
            else:
                auc_scores[task] = roc_auc_score(all_labels[task], all_preds[task])
                
        return auc_scores
    