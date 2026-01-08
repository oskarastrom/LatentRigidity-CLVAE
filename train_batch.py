import torch

import os
import tqdm
from torchmetrics import AUROC

from src.models import Image_CLVAE, EarlyStopping
from src.config import Config
from src.dataloader import get_dataloaders


def train(args, dls, data_info, save=False):
    device = 'cuda'
    model = model = Image_CLVAE(
        input_shape= [data_info['channels'], data_info['height'], data_info['width']], 
        nbr_classes= len(args.known_classes), 
        conv_layers= args.conv_layers, 
        linear_layers= args.linear_layers, 
        learned_dim= args.z_dim - args.fixed_dim, 
        fixed_dim= args.fixed_dim, 
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_marker_validation = EarlyStopping(patience=args.epochs, delta=0)
    lr_step = torch.optim.lr_scheduler.StepLR(optimizer, gamma=args.lr_gamma, step_size=args.lr_step)


    os.makedirs(args.base_path(), exist_ok=True)

    epochs = args.epochs
    batch_size = args.batch_size
    nbr_classes = len(args.known_classes)

    history = {
        'train': [],
        'val': [],
        'test': [],
        'auc': [],
    }

    def eval_batch(X, y_str):
        y = torch.tensor([int(c.split("_")[-1]) for c in y_str])
        y_cat = torch.zeros((y.shape[0], nbr_classes))
        for i in range(nbr_classes):
            y_cat[y == i, i] = 1
        y_cat = y_cat.to(device)

        recon_batch, mu, logvar = model(X)
        loss, loss_sep = model.criterion(X, y_cat, recon_batch, mu, logvar, [args.loss_rec, args.loss_KL, args.loss_cluster, args.loss_radial])

        return loss, loss_sep, mu, logvar

    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:

        if epoch % 5 == 0: torch.save(model, args.base_path() + f"/checkpoint_epoch_{epoch}.ckpt")
        
        # Training Step
        y_count = 0
        train_loss = 0
        model.train()
        train_mu = []
        train_logvar = []
        for X, y_str in dls['known']['train']:
            X = X.to(device)
            optimizer.zero_grad()
            loss, loss_sep, mu, logvar = eval_batch(X, y_str)
            train_mu.append(mu.detach().cpu())
            train_logvar.append(logvar.detach().cpu()), [args.loss_rec, args.loss_KL, args.loss_cluster, args.loss_radial]
            loss.backward()
            train_loss += loss_sep * X.size(0)
            optimizer.step()
            y_count += len(y_str)
        train_loss /= y_count
        
        
        # Validation Check
        y_count = 0
        val_loss = 0
        model.eval()
        for X, y_str in dls['known']['validation']:
            X = X.to(device)
            optimizer.zero_grad()
            loss, loss_sep, mu, logvar = eval_batch(X, y_str)
            val_loss += loss_sep * X.size(0)
            y_count += len(y_str)
        val_loss /= y_count
        
        
        # Test Check
        y_count = 0
        test_loss = 0
        model.eval()

        test_mu, test_logvar, anom_class = [], [], []
        for X, y_str in dls['test']:
            X = X.to(device)
            optimizer.zero_grad()
            loss, loss_sep, mu, logvar = eval_batch(X, y_str)
            test_mu.append(mu.detach().cpu())
            test_logvar.append(logvar.detach().cpu())
            anom_class.append(torch.tensor([int(c.split("_")[-1]) if c[0] == "k" else -1 for c in y_str]))
            test_loss += loss_sep * X.size(0)
            y_count += len(y_str)
        test_loss /= y_count
        test_mu = torch.cat(test_mu)
        test_logvar = torch.cat(test_logvar)
        anom_class = torch.cat(anom_class)

        lr_step.step()

        kl_div = torch.zeros(nbr_classes, test_mu.shape[0])
        for c_idx in range(nbr_classes):
            cm = model.get_cluster_means()[c_idx, :].cpu().detach()
            Loss_KL = -0.5 * torch.sum(1 + test_logvar - (test_mu - cm).pow(2) - test_logvar.exp(), dim=1)
            kl_div[c_idx, :] = -Loss_KL.log()
        prob = torch.softmax(kl_div, dim=0).max(0)[0]
        test_auc = AUROC('binary')(prob, anom_class != -1)
        
        
        pbar.set_description(f'Epoch {epoch+1}, Train Loss: {train_loss[0]:.4f}, Val Loss: {val_loss[0]:.4f}, Test Loss: {test_loss[0]:.4f}, Test AUC: {test_auc:.4f}')
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['test'].append(test_loss)
        history['auc'].append(test_auc)

        if model_marker_validation is not None:
            model_marker_validation(val_loss[0], model)

    if save:
        torch.save(model, args.base_path() + "/checkpoint_last.ckpt")
        model_marker_validation.load_best_model(model)
        torch.save(model, args.base_path() + "/checkpoint_best.ckpt")
        info = {
            'best_epoch_val': 100-model_marker_validation.counter,
            'history': history
        }
        torch.save(info, args.base_path() + "/info.pt")
        
    return model, history



if __name__ == "__main__":
    device = "cuda"
    for dataset in  ['mnist', 'cifar10', 'svhn', 'tiny_imagenet', 'cifar+10', 'cifar+50']:
        for split_num in range(1):
            new_dataset = True
            for fixed_dims in [0, 8, 64, 128]:
                args = Config(dataset=dataset, split_num=split_num, latent_dims=128, fixed_dims=fixed_dims)
                if new_dataset:
                    dls, data_info = get_dataloaders(args)
                    new_dataset = False
                model, history = train(args, dls, data_info, save=True)
        