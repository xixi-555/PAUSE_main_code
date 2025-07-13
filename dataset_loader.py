import os.path
import numpy as np
import scipy.io as sio
from scipy import sparse
import sklearn.preprocessing as skp
import torch
from utils import *


def load_mat(args):
    """
    Load multi-view features and ground-truth labels from .mat files based on the specified dataset.
    Supports various normalization strategies for preprocessing.

    Args:
        args: Argument namespace containing dataset name, path, normalization type, etc.

    Returns:
        data_X (list of np.ndarray): List of view-specific feature matrices.
        label_y (np.ndarray): Ground-truth labels.
        label_y_view2 (np.ndarray): Duplicate label array (e.g., for view 2).
        id1 (np.ndarray): Sample indices for view 1.
        id2 (np.ndarray): Sample indices for view 2.
    """
    data_X = []
    label_y = None

    if args.dataset == "WIKI":
        mat = sio.loadmat(os.path.join(args.data_path, "WIKI.mat"))
        data_X.append(mat["Img"].astype("float32"))
        data_X.append(mat["Txt"].astype("float32"))
        label_y = np.squeeze(mat["label"])

    elif args.dataset == "NoisyMNIST":
        mat = sio.loadmat(os.path.join(args.data_path, "NoisyMNIST.mat"))
        data_X.append(mat["X1"].astype("float32"))
        data_X.append(mat["X2"].astype("float32"))
        label_y = np.squeeze(mat["Y"])

    elif args.dataset == "MNISTUSPS":
        mat = sio.loadmat(os.path.join(args.data_path, "MNIST-USPS.mat"))
        data_X.append(mat["X1"].astype("float32"))
        data_X.append(mat["X2"].astype("float32"))
        label_y = np.squeeze(mat["Y"])

    elif args.dataset == "NUSWIDE":
        mat = sio.loadmat(os.path.join(args.data_path, "nuswide_deep_2_view.mat"))
        data_X.append(mat["Img"])  # Visual modality
        data_X.append(mat["Txt"])  # Text modality
        label_y = np.squeeze(mat["label"].T)

    elif args.dataset == "CUB":
        mat = sio.loadmat(os.path.join(args.data_path, "CUB.mat"))
        data_X.append(mat["X"][0][0].astype("float32"))
        data_X.append(mat["X"][0][1].astype("float32"))
        label_y = np.squeeze(mat["gt"])

    else:
        raise KeyError(f"Unknown dataset: {args.dataset}")

    # Apply selected normalization
    if args.data_norm == "standard":
        for i in range(args.n_views):
            data_X[i] = skp.scale(data_X[i])
    elif args.data_norm == "l2-norm":
        for i in range(args.n_views):
            data_X[i] = skp.normalize(data_X[i])
    elif args.data_norm == "min-max":
        for i in range(args.n_views):
            data_X[i] = skp.minmax_scale(data_X[i])

    label_y_view2 = label_y.copy()
    id1 = np.arange(data_X[0].shape[0])
    id2 = id1.copy()
    args.n_sample = data_X[0].shape[0]

    return data_X, label_y, label_y_view2, id1, id2


def get_pseudo_label_path(args):
    """
    Construct the file path for pseudo labels generated during the warm-up phase.

    Returns:
        str: Full file path to the .npy pseudo label file.
    """
    return os.path.join(
        args.output_dir, f"{args.dataset}_warmup_pseudo_labels_seed{args.seed}.npy"
    )


def load_pseudo_labels(pseudo_label_path, args):
    """
    Load pseudo labels (usually generated after warm-up) from a .npy file.

    Args:
        pseudo_label_path (str): Path to the pseudo label file.
        args: Global argument namespace (used for path configuration if needed).

    Returns:
        np.ndarray: Array of pseudo labels.
    """
    full_path = pseudo_label_path
    pseudo_labels = np.load(full_path)
    return pseudo_labels


def load_dataset(args, use_pseudo_labels=False):
    """
    Load multi-view dataset and construct a PyTorch dataset object.

    Args:
        args: Argument namespace containing dataset configuration.
        use_pseudo_labels (bool): Whether to include pseudo labels (e.g., in fine-tuning).

    Returns:
        dataset (torch.utils.data.Dataset): The wrapped multi-view dataset object.
        input_dims (list[int]): List of feature dimensions per view.
    """
    data, label1, label2, id1, id2 = load_mat(args)
    _ = len(np.unique(label1))  # class count (can be used if needed)

    if use_pseudo_labels:
        pseudo_labels_path = get_pseudo_label_path(args)
        pseudo_labels = load_pseudo_labels(pseudo_labels_path, args)
        dataset = NewMultiviewDataset(args.n_views, data, label1, label2, id1, id2, pseudo_label=pseudo_labels)
    else:
        dataset = MultiviewDataset(args.n_views, data, label1, label2, id1, id2)

    return dataset, [data[0].shape[1], data[1].shape[1]]


class MultiviewDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for multi-view data without pseudo labels (typically used during warm-up).
    """

    def __init__(self, n_views, data_X, label1, label2, id1, id2):
        super(MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.label1 = label1 - np.min(label1)
        self.label2 = label2 - np.min(label2)
        self.id1 = id1
        self.id2 = id2

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = [torch.tensor(self.data[i][idx].astype("float32")) for i in range(self.n_views)]
        label1 = torch.tensor(self.label1[idx], dtype=torch.long)
        label2 = torch.tensor(self.label2[idx], dtype=torch.long)
        id1 = torch.tensor(self.id1[idx], dtype=torch.long)
        id2 = torch.tensor(self.id2[idx], dtype=torch.long)
        return idx, data, label1, label2, id1, id2


class NewMultiviewDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for multi-view data with pseudo labels (used after warm-up phase).
    """

    def __init__(self, n_views, data_X, label1, label2, id1, id2, pseudo_label):
        super(NewMultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.label1 = label1 - np.min(label1)
        self.label2 = label2 - np.min(label2)
        self.id1 = id1
        self.id2 = id2
        self.pseudo_label = pseudo_label

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = [torch.tensor(self.data[i][idx].astype("float32")) for i in range(self.n_views)]
        label1 = torch.tensor(self.label1[idx], dtype=torch.long)
        label2 = torch.tensor(self.label2[idx], dtype=torch.long)
        id1 = torch.tensor(self.id1[idx], dtype=torch.long)
        id2 = torch.tensor(self.id2[idx], dtype=torch.long)
        pseudo_label = torch.tensor(self.pseudo_label[idx], dtype=torch.long)
        return idx, data, label1, label2, id1, id2, pseudo_label

