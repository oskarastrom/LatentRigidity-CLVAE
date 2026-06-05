##################################################
# Imports
##################################################

import pytorch_lightning as pl
import os


def get_callbacks(args):
    callbacks = []

    # Model checkpoint
    model_checkpoint_clbk = pl.callbacks.ModelCheckpoint(
        dirpath=None,
        filename='best',
        monitor='validation_acc',
        save_last=True,
        mode='max',
    )
    callbacks += [model_checkpoint_clbk]
    return callbacks

def get_logger(args):
    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), 'tmp'),
        name=args.dataset,
    )
    return tb_logger

def generate_save_name(folder, save_path):
    A = os.listdir(folder)
    A = [a for a in A if a.startswith(save_path) and a.endswith(".ckpt")]
    if save_path + ".ckpt" not in A: 
        return folder + "/" + (save_path + ".ckpt")
    else:
        A = [a[len(save_path):] for a in A]
        A = [a for a in A if "_" in a]
        if len(A) == 0:
            return folder + "/" + (save_path + "_" + str(2) + ".ckpt")
        else:
            counts = [a.split("_")[-1][:-5] for a in A]
            counts = [int(a) for a in counts if a.isdigit()]
            max_count = max(counts)
            return folder + "/" + save_path + "_" + str(max_count + 1) + ".ckpt"
        
        
import torch

# function to spread points on a hypersphere using energy minimization
def scatter_on_hypersphere(N, dim, steps=1000, lr=1e-2, device='cpu'):

    if dim == 0 or N == 0:
        return torch.zeros((N, dim))
    
    # initialize points
    x = torch.randn(N, dim, device=device)
    x = x / x.norm(dim=1, keepdim=True)
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        # calc distances and forces etc
        x_normalized = x / x.norm(dim=1, keepdim=True)
        diffs = x_normalized[:, None, :] - x_normalized[None, :, :]
        dists = torch.norm(diffs, dim=-1) + torch.eye(N)

        loss = torch.sum(1.0 / dists) - torch.sum(torch.diag(1.0 / dists))

        loss.backward()
        optimizer.step()

    x = x.detach()
    x = x / x.norm(dim=1, keepdim=True)
    return x