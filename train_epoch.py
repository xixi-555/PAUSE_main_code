import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from utils import *


# Warm-up stage: train with inter-view InfoNCE, intra-view InfoNCE, and reconstruction loss
def train_warmup_epoch_with_three_loss(model, train_loader, optimizer, epoch, device, args):
    model.train()
    running_loss = 0.0
    mse_loss_fn = torch.nn.MSELoss()

    for i, (idx, views, label1, label2, id1, id2) in enumerate(train_loader):
        views = [v.to(device) for v in views]

        optimizer.zero_grad()
        z, reconstructed = model(*views)

        emb_i, emb_j = z[0].to(device), z[1].to(device)
        emb_i_argumentation = data_augmentation(emb_i, args)
        emb_j_argumentation = data_augmentation(emb_j, args)

        batch_size = emb_i.size(0)
        info_nce_loss_fn = Info_Nce_Loss(batch_size=batch_size, device=device, temperature=args.temperature)

        # Intra-view InfoNCE loss: emb vs its augmentation
        intra_info_nce_loss = info_nce_loss_fn(emb_i, emb_i_argumentation) + info_nce_loss_fn(emb_j, emb_j_argumentation)

        # Inter-view InfoNCE loss: emb_i vs emb_j
        inter_info_nce_loss = info_nce_loss_fn(emb_i, emb_j)

        # Reconstruction loss for both views
        reconstruction_loss = sum(mse_loss_fn(reconstructed[j], views[j]) for j in range(model.n_views))

        # Total loss
        total_loss = args.alpha * inter_info_nce_loss + reconstruction_loss + args.beta * intra_info_nce_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(train_loader)


# Fine-tuning stage: train with cross-view Universum loss, intra-view Universum loss, and reconstruction loss
def train_universum_epoch_new_with_three_loss(model, train_loader, optimizer, epoch, device, args):
    model.train()
    running_loss = 0.0
    mse_loss_fn = torch.nn.MSELoss()

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args.epochs}]", unit="batch", dynamic_ncols=True, disable=True)

    for i, (idx, views, label1, label2, id1, id2, pseudo_labels) in enumerate(progress_bar):
        views = [v.to(device) for v in views]
        pseudo_labels = torch.tensor(pseudo_labels).to(device)

        optimizer.zero_grad()
        z, reconstructed = model(*views)

        emb_i, emb_j = z[0].to(device), z[1].to(device)
        emb_i_argumentation = data_augmentation(emb_i, args)
        emb_j_argumentation = data_augmentation(emb_j, args)

        # Reconstruction loss
        reconstruction_loss = sum(mse_loss_fn(reconstructed[j], views[j]) for j in range(model.n_views))

        # Inter-view Universum loss
        inter_uni_loss = compute_single_uni_loss_new(
            encoded_view1=emb_i,
            encoded_view2=emb_j,
            pseudo_labels=pseudo_labels,
            temperature=0.5
        )

        # Intra-view Universum loss (with augmentations)
        intra_uni_loss = (
            compute_single_uni_loss_new(emb_i, emb_i_argumentation, pseudo_labels, temperature=0.5) +
            compute_single_uni_loss_new(emb_i, emb_j_argumentation, pseudo_labels, temperature=0.5)
        )

        # Total loss
        gamma2 = args.gamma / args.ratio
        total_loss = reconstruction_loss + args.gamma * inter_uni_loss + gamma2 * intra_uni_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(train_loader)
