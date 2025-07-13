import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import datetime
import os
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import torch
from munkres import Munkres
import random

# -------------------- seed --------------------

def set_random_seed(seed):
    """
    Set global random seed for reproducibility across random, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------- loss --------------------

class Info_Nce_Loss(nn.Module):
    """
    Standard InfoNCE loss for contrastive learning.
    Encourages paired samples to be close, while pushing others apart.
    """

    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def batch_get_universum(encoded_view1, encoded_view2, pseudo_labels, lambda_=0.5):
    """
    Generate universum centers for each sample via Mixup with other-class centroids.

    Args:
        encoded_view1: (N, D) feature matrix from primary view.
        encoded_view2: (N, D) feature matrix from auxiliary view.
        pseudo_labels: (N,) pseudo-labels for samples.
        lambda_: Mixing ratio for universum generation.

    Returns:
        (N, D) tensor of universum centers.
    """
    N, D = encoded_view1.shape
    unique_labels = torch.unique(pseudo_labels)
    cluster_means = torch.stack([
        encoded_view2[pseudo_labels == label].mean(dim=0)
        for label in unique_labels
    ])

    universum_samples = []
    for i in range(N):
        label = pseudo_labels[i]
        other_means = cluster_means[unique_labels != label]
        mixed = [lambda_ * encoded_view1[i] + (1 - lambda_) * mean for mean in other_means]
        universum_samples.append(torch.stack(mixed).mean(dim=0))

    return torch.stack(universum_samples)


def compute_single_uni_loss_new(encoded_view1, encoded_view2, pseudo_labels, temperature=0.5):
    """
    Compute UniLoss: encourage same-class similarity while pushing away universum and different-class pairs.

    Args:
        encoded_view1: Features from primary view.
        encoded_view2: Features from auxiliary view.
        pseudo_labels: Sample-level pseudo labels.
        temperature: Softmax temperature.

    Returns:
        Scalar UniLoss.
    """
    universum_samples = batch_get_universum(encoded_view1, encoded_view2, pseudo_labels)

    pos_mask = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)
    neg_mask = ~pos_mask

    sim_12 = F.cosine_similarity(encoded_view1.unsqueeze(1), encoded_view2.unsqueeze(0), dim=2)
    sim_1u = F.cosine_similarity(encoded_view1.unsqueeze(1), universum_samples, dim=2)
    sim_2u = F.cosine_similarity(encoded_view2.unsqueeze(1), universum_samples, dim=2)

    pos_exp = torch.exp(sim_12 / temperature) * pos_mask.float()
    numerator = torch.sum(pos_exp, dim=1)

    neg_exp = torch.exp(sim_12 / temperature) * neg_mask.float()
    denominator = torch.sum(neg_exp, dim=1)
    denominator += torch.sum(torch.exp(sim_1u / temperature), dim=1)
    denominator += torch.sum(torch.exp(sim_2u / temperature), dim=1)

    return (-torch.log(numerator / denominator)).mean()


def compute_clustering_loss(embeddings, cluster_centers, pseudo_labels, temperature=1.0):
    """
    Compute clustering loss via cross-entropy over distance-based logits.

    Args:
        embeddings: (N, D) sample embeddings.
        cluster_centers: (K, D) cluster centroids.
        pseudo_labels: Ground truth or predicted labels.
        temperature: Softmax temperature.

    Returns:
        Scalar clustering loss.
    """
    distance = torch.cdist(embeddings, cluster_centers, p=2)
    logits = distance / temperature
    return F.cross_entropy(logits, pseudo_labels)

# -------------------- evaluation --------------------

def save_pseudo_labels(pseudo_labels, pseudo_labels_path):
    """
    Save pseudo labels as a .npy file to the specified path.
    """
    save_path = Path(pseudo_labels_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, pseudo_labels)



def get_pseudo_label_path(args):
    """
    Generate path for saving/loading pseudo labels based on dataset and seed.
    """
    return os.path.join(args.output_dir, f"{args.dataset}_warmup_pseudo_labels_seed{args.seed}.npy")


def load_pseudo_labels(pseudo_label_path, args):
    """
    Load pseudo labels from a .npy file.
    """
    full_path = pseudo_label_path
    return np.load(full_path)


def evaluate_with_seed(model, data_loader, device, n_clusters, seed):
    """
    Evaluate model performance via clustering metrics (ACC, NMI, ARI).
    Extract features, cluster with KMeans, and align labels using Munkres.
    """
    model.eval()
    features = []
    true_labels = []

    with torch.no_grad():
        for _, views, label1, *_ in data_loader:
            views = [view.to(device) for view in views]
            z, _ = model(*views)
            features.append(torch.cat(z, dim=1).cpu().numpy())
            true_labels.append(label1.cpu().numpy())

    features = np.vstack(features)
    true_labels = np.concatenate(true_labels)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_assignments = kmeans.fit_predict(features)

    nmi = metrics.normalized_mutual_info_score(true_labels, cluster_assignments)
    ari = metrics.adjusted_rand_score(true_labels, cluster_assignments)
    aligned_preds = get_y_preds(true_labels, cluster_assignments, n_clusters)
    acc = metrics.accuracy_score(true_labels, aligned_preds)

    return acc, nmi, ari, cluster_assignments


def calculate_cost_matrix(C, n_clusters):
    """
    Compute cost matrix for Munkres algorithm based on the confusion matrix.
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    """
    Convert Munkres index pairs to label mapping array.
    """
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Align clustering assignments with ground truth using the Munkres algorithm.
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    label_mapping = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments -= np.min(cluster_assignments)

    y_pred = label_mapping[cluster_assignments]
    return y_pred

# -------------------- pseudo-label --------------------

def save_pseudo_labels(pseudo_labels, pseudo_labels_path):
    """
    Save pseudo labels to a .npy file at the specified path.
    """
    save_path = Path(pseudo_labels_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, pseudo_labels)


def get_pseudo_label_path(args):
    """
    Generate file path for saving/loading pseudo labels based on dataset and seed.
    """
    return os.path.join(args.output_dir, f"{args.dataset}_warmup_pseudo_labels_seed{args.seed}.npy")


def load_pseudo_labels(pseudo_label_path, args):
    """
    Load pseudo labels from a .npy file.
    """
    full_path = pseudo_label_path
    return np.load(full_path)


# -------------------- logging --------------------

def save_checkpoint(model, optimizer, best_metrics, args, current_epoch, state_logger, phase):
    """
    Save model checkpoint including epoch, model state, optimizer state, and best metrics.
    """
    checkpoint_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{phase}_seed{args.seed}_checkpoint_epoch_{current_epoch}.pth"
    )
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metrics': best_metrics,
    }, checkpoint_path)
    state_logger.write(f"Checkpoint saved to {checkpoint_path}\n")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state from a saved checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_metrics = checkpoint['best_metrics']

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resumed from epoch {epoch + 1}")

    return model, optimizer, epoch, best_metrics


class FileLogger:
    """
    A simple logger that writes timestamped logs to both file and console.
    """
    def __init__(self, output_file):
        self.output_file = output_file

    def write(self, msg, p=True):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        log_message = f"[{current_time}] {msg}"
        with open(self.output_file, mode="a", encoding="utf-8") as log_file:
            log_file.writelines(log_message + "\n")
        if p:
            print(log_message)


def log_warmup_train_loss(state_logger, current_epoch, total_epochs, train_loss):
    """
    Log training loss during the warm-up stage.
    """
    state_logger.write(
        f"Epoch [{current_epoch}/{total_epochs}] - Train Loss: {train_loss:.4f} - Warmup Training...\n"
    )


def log_universum_train_loss(state_logger, current_epoch, total_epochs, train_loss):
    """
    Log training loss during the universum stage.
    """
    state_logger.write(
        f"Epoch [{current_epoch}/{total_epochs}] - Train Loss: {train_loss:.4f} - Universum Training...\n"
    )


def log_evaluation(state_logger, current_epoch, total_epochs, nmi, acc, ari):
    """
    Log evaluation metrics for current epoch.
    """
    state_logger.write(
        f"Evaluation - Epoch {current_epoch} - ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}\n"
    )


# -------------------- data_augmentation --------------------

def data_augmentation(features, args):
    """
    Apply the selected data augmentation strategy based on args.
    """
    if args.augmentation_strategy == "gaussian_noise":
        features = add_gaussian_noise(features, mean=args.noise_mean, std=args.noise_std)

    elif args.augmentation_strategy == "feature_dropout":
        features = feature_dropout(features, drop_rate=args.feature_drop_rate)

    elif args.augmentation_strategy == "selective_dropout":
        features = selective_dropout(features, drop_rate=args.feature_drop_rate, importance_scores=None)

    return features


def add_gaussian_noise(features, mean=0.0, std=0.1):
    """
    Add Gaussian noise to input features.
    """
    noise = torch.randn_like(features) * std + mean
    return features + noise


def feature_dropout(features, drop_rate=0.4):
    """
    Randomly zero out a portion of feature values.
    """
    mask = torch.rand_like(features) > drop_rate
    return features * mask


def selective_dropout(features, drop_rate=0.2, importance_scores=None):
    """
    Drop least important features based on importance scores.
    If no scores are provided, all features are treated equally.
    """
    if importance_scores is None:
        importance_scores = torch.ones(features.size(1), device=features.device)

    sorted_indices = torch.argsort(importance_scores)
    num_to_drop = int(features.size(1) * drop_rate)
    drop_indices = sorted_indices[:num_to_drop]

    mask = torch.ones(features.size(1), device=features.device)
    mask[drop_indices] = 0

    return features * mask

