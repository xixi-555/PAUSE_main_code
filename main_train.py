from train_epoch import *
from dataset_loader import *
from model import *
import warnings
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import torch
import numpy as np
import random
import datetime
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training Multi-View Contrastive Clustering Model")

    # Configuration file path
    parser.add_argument("--config_file", type=str, help="Path to YAML configuration file")

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Temperature parameter for contrastive loss
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature parameter for contrastive loss")

    # Number of clusters for K-means
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for K-means")

    # Number of views
    parser.add_argument("--n_views", type=int, default=2, help="Number of views")

    # Dropout rate in model layers
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate in the model layers")

    # Weight decay for optimizer regularization
    parser.add_argument("--weight_decay", type=float, default=0.00005, help="Weight decay for optimizer regularization")

    # Weight decay for warmup optimizer regularization
    parser.add_argument("--warmup_weight_decay", type=float, default=0.00001,
                        help="Weight decay for warmup optimizer regularization")

    # Batch size for training
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")

    # Normalization type for dataset
    parser.add_argument(
        "--data_norm", type=str, default="min-max", choices=["standard", "min-max", "l2-norm"],
        help="Normalization type for dataset"
    )

    # Path to dataset folder
    parser.add_argument("--data_path", type=str, default="./dataset", help="Path to dataset folder")

    # Dataset to use for training
    parser.add_argument(
        "--dataset", type=str, default="WIKI", choices=["WIKI", "NoisyMNIST", "MNISTUSPS", "NUSWIDE", "CUB"],
        help="Dataset to use for training"
    )

    # Number of training epochs
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")

    # Lambda parameter for UniLoss
    parser.add_argument("--lambda_", type=float, default=0.5, help="Lambda parameter for UniLoss")

    # Number of warmup training epochs
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup training epochs")

    # Learning rate for the optimizer
    parser.add_argument("--lr", type=float, default=0.00001, metavar="LR", help="Learning rate for the optimizer")

    # Warmup learning rate for the optimizer
    parser.add_argument("--warmup_lr", type=float, default=0.001, metavar="LR",
                        help="WARMUP_Learning rate for the optimizer")

    # Frequency of printing training status
    parser.add_argument("--print_freq", type=int, default=20, help="Frequency of printing training status")

    # Optimizer to use for training
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimizer to use for training")

    # Device to train on: 'cuda' or 'cpu'
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to train on: 'cuda' or 'cpu'")

    # Weight for Universum loss
    parser.add_argument('--gamma', type=float, default=0.05, help='Weight for inter Universum loss')

    # Weight for intra InfoNCE loss
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight for intra InfoNCE loss')

    # Weight for inter InfoNCE loss
    parser.add_argument('--beta', type=float, default=0.2, help='Weight for inter InfoNCE loss')

    # Ratio for intra-view and inter-view losses
    parser.add_argument('--ratio', type=float, default=2.5, help='Weight for intra Universum loss * ratio = Weight for inter Universum loss')

    # Directory to save logs
    parser.add_argument("--output_log_dir", type=str, default="./logs", help="Directory to logs")

    # Directory to save dataset and checkpoints
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to dataset and checkpoints")

    # Augmentation strategy
    parser.add_argument(
        "--augmentation_strategy", type=str, choices=["feature_dropout", "selective_dropout"],
        default="feature_dropout", help="Choice of augmentation strategy"
    )

    # Feature dropout rate
    parser.add_argument("--feature_drop_rate", type=float, default=0.4,
                        help="Feature Dropout rate: proportion of features to drop")

    # Whether to save checkpoints during training
    parser.add_argument("--save_ckpt", action="store_true", help="Whether to save checkpoints during training")

    # Epoch to start training from (for resuming training)
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start training from")

    # Whether to print status every epoch
    parser.add_argument("--print_this_epoch", action="store_true", help="Whether to print status every epoch")

    # Number of workers for data loading
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

    # Recognition model layer dimensions
    parser.add_argument(
        '--recognition_model_dims', default=[[784, 1024, 1024, 256], [784, 1024, 1024, 256]], type=list,
        help='Recognition model layer dimensions in the form of a list of lists.'
    )

    # Generative model layer dimensions
    parser.add_argument(
        '--generative_model_dims', default=[[256, 1024, 1024, 784], [256, 1024, 1024, 784]], type=list,
        help='Generative model layer dimensions in the form of a list of lists.'
    )

    # Whether to use dropout in the model
    parser.add_argument('--use_dropout', default=True, type=bool, help='Whether to use dropout in the model.')

    # Activation function used in the adaption layer of the recognition model
    parser.add_argument(
        '--activation', default='relu', type=str, choices=['none', 'relu', 'tanh'],
        help='Activation function used in the adaption layer of the recognition model.'
    )

    return parser


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args_and_config(seed):
    parser = get_args_parser()
    args = parser.parse_args()

    if not args.config_file:
        args.config_file = os.path.join("config", f"{args.dataset}.yaml")

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config not found: {args.config_file}")

    with open(args.config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    args = vars(args)
    args.update(yaml_config)
    args['seed'] = seed
    args = argparse.Namespace(**args)

    return args

def train_with_seed(seed):
    # Load config and parse arguments
    args = parse_args_and_config(seed)
    set_seed(args.seed)

    os.makedirs(args.output_log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log_filename = os.path.join(
        args.output_log_dir,
        f"log_main_{args.dataset}_seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    state_logger = FileLogger(log_filename)
    state_logger.write(f"Starting Main Training for dataset: {args.dataset}, seed: {args.seed}\n")
    #state_logger.write(f"Parameters: {vars(args)}\n")

    # -------------------- Step 1: Warm-up Training --------------------
    dataset, _ = load_dataset(args, use_pseudo_labels=False)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultiViewAutoencoderWithClustering(
        n_views=args.n_views,
        recognition_model_dims=args.recognition_model_dims,
        generative_model_dims=args.generative_model_dims,
        temperature=args.temperature,
        n_clusters=args.n_clusters,
        drop_rate=args.drop_rate,
        args=args
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.warmup_lr, weight_decay=args.warmup_weight_decay)
    warmup_losses, universum_losses = [], []
    final_metrics = {"acc": None, "nmi": None, "ari": None}


    with tqdm(range(1, args.warmup_epochs + 1), desc="Warm-up Training", disable=True) as progress_bar1:
        for current_epoch in progress_bar1:
            model.train()
            train_loss = train_warmup_epoch_with_three_loss(model, train_loader, optimizer, current_epoch, args.device, args)
            warmup_losses.append(train_loss)

            if current_epoch % args.print_freq == 0 or current_epoch == args.warmup_epochs:
                model.eval()
                with torch.no_grad():
                    acc, nmi, ari, pred_labels = evaluate_with_seed(model, test_loader, args.device, args.n_clusters, args.seed)
                    log_evaluation(state_logger, current_epoch, args.warmup_epochs, nmi, acc, ari)

                if current_epoch == args.warmup_epochs:
                    pseudo_labels_path = os.path.join(args.output_dir, f"{args.dataset}_warmup_pseudo_labels_seed{args.seed}.npy")
                    save_pseudo_labels(pred_labels, pseudo_labels_path)


    # -------------------- Step 2: Universum Training --------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset_with_pseudo, _ = load_dataset(args, use_pseudo_labels=True)

    train_loader = torch.utils.data.DataLoader(dataset_with_pseudo, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    universum_epochs = args.epochs - args.warmup_epochs

    with tqdm(range(1, universum_epochs + 1), desc="Universum Training", disable=True) as progress_bar2:
        for epoch in progress_bar2:
            current_epoch = args.warmup_epochs + epoch
            model.train()
            train_loss = train_universum_epoch_new_with_three_loss(model, train_loader, optimizer, current_epoch, args.device, args)
            universum_losses.append(train_loss)

            if current_epoch % args.print_freq == 0 or current_epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    acc, nmi, ari, _ = evaluate_with_seed(model, test_loader, args.device, args.n_clusters, args.seed)
                    log_evaluation(state_logger, current_epoch, args.epochs, nmi, acc, ari)
                    final_metrics = {"acc": acc, "nmi": nmi, "ari": ari}

    state_logger.write(
        f"Final Metrics - ACC: {final_metrics['acc']:.4f}, "
        f"NMI: {final_metrics['nmi']:.4f}, ARI: {final_metrics['ari']:.4f}\n"
    )

    return final_metrics


def run_multiple_seeds(num_seeds=5, initial_seed=42, step=1):
    """
    Train with multiple seeds and compute the mean/std of final metrics.
    """
    seeds = [initial_seed + step * i for i in range(num_seeds)]

    initial_args = parse_args_and_config(seeds[0])
    dataset_name = initial_args.dataset
    output_log_dir = initial_args.output_log_dir
    os.makedirs(output_log_dir, exist_ok=True)

    current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_log_filename = os.path.join(output_log_dir, f"summary_metrics_{dataset_name}_{current_time_str}.txt")

    all_metrics = {"acc": [], "nmi": [], "ari": []}
    per_seed_results = []

    with open(summary_log_filename, "w") as f_summary:
        f_summary.write(f"===== Summary for [{dataset_name}] at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n\n")
        f_summary.write("===== Parameters =====\n")
        for k, v in vars(initial_args).items():
            f_summary.write(f"{k}: {v}\n")
        f_summary.write("\n")

        for i, seed in enumerate(seeds, 1):
            print(f"\n===== Seed {seed} ({i}/{num_seeds}) =====\n")
            metrics = train_with_seed(seed)

            for k in all_metrics:
                all_metrics[k].append(metrics[k])
            per_seed_results.append((seed, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), metrics))

            f_summary.write(
                f"Seed {seed} completed with ACC: {metrics['acc']:.4f}, "
                f"NMI: {metrics['nmi']:.4f}, ARI: {metrics['ari']:.4f}\n"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in all_metrics.items()}

        f_summary.write("\n===== Overall Metrics =====\n")
        for k in all_metrics:
            f_summary.write(f"{k.upper()}: Mean = {mean_metrics[k]:.4f}, Std = {std_metrics[k]:.4f}\n")

    print("===== Summary =====")
    for k in all_metrics:
        print(f"{k.upper()}: Mean = {mean_metrics[k]:.4f}, Std = {std_metrics[k]:.4f}")
    print(f"Summary saved to {summary_log_filename}")

if __name__ == "__main__":
    run_multiple_seeds(num_seeds=5, initial_seed=42)
