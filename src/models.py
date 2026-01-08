from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from enum import Enum
import copy
from utils import scatter_on_hypersphere

class LatentClustering(Enum):
    ZEROED = 1
    FIXED = 2
    LEARNED = 3
    FREE = 4
    DISC = 5
    CYLINDER = 6
    TRACTORBEAM = 7
    
class Image_CLVAE(nn.Module):
    def __init__(self, 
            input_shape=64, 
            conv_layers=[(64, 4, 2, 1)],
            linear_layers=[64],
            nbr_classes=9,
            zeroed_dim=0, 
            fixed_dim=2, 
            learned_dim=0, 
            free_dim=0, 
        ):
        super(Image_CLVAE, self).__init__()
        self.dim_reduction = None
        self.device = "cpu"
        self.input_shape = input_shape
        self.zeroed_dim = zeroed_dim
        self.fixed_dim = fixed_dim
        self.learned_dim = learned_dim
        self.free_dim = free_dim
        
        self.latent_dim = zeroed_dim + fixed_dim + learned_dim + free_dim
        self.generate_cluster_means(nbr_classes)
        self.generate_layers(conv_layers, linear_layers)
        
    def generate_layers(self, conv_layers, linear_layers):
        # Encoder layers
        self.encoder_conv = self._make_encoder_conv(conv_layers)
        self.encoder_linear = self._make_encoder_linear(linear_layers)
        self.mu = nn.Linear(self.prelatent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.prelatent_dim, self.latent_dim)
        
        # Decoding Layers
        self.decoder_linear = self._make_decoder_linear(linear_layers)
        self.decoder_conv = self._make_decoder_conv(conv_layers)
        
    
    def generate_cluster_means(self, nbr_classes):
        self.nbr_classes = nbr_classes

        self.zeroed_mean = torch.zeros((nbr_classes, self.zeroed_dim), dtype=torch.float32)
        
        self.fixed_mean = scatter_on_hypersphere(nbr_classes, self.fixed_dim)
        #zeroed = np.random.choice(nbr_classes, 3, replace=False)
        #self.fixed_mean = torch.zeros((nbr_classes, self.fixed_dim), dtype=torch.float32)
        #self.fixed_mean[zeroed, :] = scatter_on_hypersphere(3, self.fixed_dim)
        
        self.learned_mean = scatter_on_hypersphere(nbr_classes, self.learned_dim)
        self.learned_mean = nn.Parameter(self.learned_mean)
        
    def get_cluster_means(self):
        return torch.concatenate([self.zeroed_mean, self.fixed_mean, self.learned_mean], dim=1)
    
    def _make_encoder_conv(self, conv_layers):
        layers = []
        in_channels = self.input_shape[0]
        for out_channels, kernel_size, stride, padding in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_encoder_linear(self, linear_layers):
        # Compute convolution output shape and size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            conv_out = self.encoder_conv(dummy)
            self.conv_out_shape = conv_out.shape[1:]  # C, H, W
            self.flat_dim = conv_out.reshape(1, -1).size(1)

        # Create Linear Layers
        encoder_linears = []
        in_dim = self.flat_dim
        for h_dim in linear_layers:
            encoder_linears.append(nn.Linear(in_dim, h_dim))
            encoder_linears.append(nn.ReLU())
            in_dim = h_dim
        self.prelatent_dim = in_dim
        return nn.Sequential(*encoder_linears)
        
    def _make_decoder_conv(self, conv_layers):
        layers = []
        reversed_layers = list(reversed(conv_layers))
        in_channels = self.conv_out_shape[0]
        for i, (_, kernel_size, stride, padding) in enumerate(reversed_layers):
            out_channels = reversed_layers[i+1][0] if i+1 < len(reversed_layers) else self.input_shape[0]
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers[-1] = nn.Sigmoid()
        return nn.Sequential(*layers)
    
    def _make_decoder_linear(self, linear_layers):
        decoder_linears = []
        reverse_linear_layers = list(reversed(linear_layers))
        in_dim = self.latent_dim
        for h_dim in reverse_linear_layers:
            decoder_linears.append(nn.Linear(in_dim, h_dim))
            decoder_linears.append(nn.ReLU())
            in_dim = h_dim
        decoder_linears.append(nn.Linear(in_dim, self.flat_dim))
        decoder_linears.append(nn.ReLU())
        return nn.Sequential(*decoder_linears)

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        h = self.encoder_linear(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_linear(z)
        h = h.reshape(h.size(0), *self.conv_out_shape)
        x = self.decoder_conv(h)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
  
  
    def fit_dim_reduction(self, basis_datasets, dims):
        print("ERROR: dimension reduction is not available on the Convolutional CLVAE")
        self.dim_reduction = None
    
    def prep_features(self, X):
        if self.dim_reduction is not None:
            X = self.dim_reduction.transform(X)
        X = X.to(self.device)
        return X
        
    def to(self, device):
        self.device = device
        self.fixed_mean = self.fixed_mean.to(device)
        self.zeroed_mean = self.zeroed_mean.to(device)
        return nn.Module.to(self, device)
  
    def criterion(self, X, Y, x_rec, mu, logvar, loss_fac):
        # Calculation of loss
        
        # Reconstruction error in the form of MSE
        Loss_MSE = F.binary_cross_entropy(x_rec, X, reduction='mean')
        Loss_MSE *= loss_fac[0]
        
        # Kullback–Leibler divergence loss, based on the cluster_mean parameters of the model
        mu_nonzero = mu[:, :-self.free_dim] if self.free_dim > 0 else mu
        logvar_nonzero = logvar[:, :-self.free_dim] if self.free_dim > 0 else logvar
        cluster_x = torch.matmul(Y, self.get_cluster_means())
        z_prior_mean = mu_nonzero - cluster_x
        Loss_KL = -0.5 * torch.sum(1 + logvar_nonzero - z_prior_mean.pow(2) - logvar_nonzero.exp())
        Loss_KL *= loss_fac[1]
        
        # Loss that forces the clusters away from each other
        Loss_cluster = torch.tensor([0], device=self.device)
        Loss_origin = torch.tensor([0], device=self.device)
        if self.learned_dim > 0:
            learnable_clusters = self.get_cluster_means()[:, self.zeroed_dim+self.fixed_dim:]
            diffs = learnable_clusters[:, None, :] - learnable_clusters[None, :, :]
            dists = torch.norm(diffs, dim=-1) + torch.eye(learnable_clusters.shape[0], device=diffs.device)
            cluster_separation = torch.sum(1.0 / dists) - torch.sum(torch.diag(1.0 / dists))
            Loss_cluster = torch.sum(cluster_separation)
            Loss_cluster *= loss_fac[2]
            
            # Loss that forces the clusters away from the origin
            origin_separation = learnable_clusters.norm(2, dim=1)
            Loss_origin = torch.sum(origin_separation - torch.log(origin_separation))
            Loss_origin *= loss_fac[3]
        
        # Return both the total loss object, but also a list of the separate losses for analysis
        total_loss = Loss_MSE + Loss_KL + Loss_cluster + Loss_origin
        return total_loss, torch.tensor([total_loss.item(), Loss_MSE.item(), Loss_KL.item(), Loss_cluster.item(), Loss_origin.item()])
   
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
   
       