import torch.nn as nn
import torch 
import torch.nn.functional as F
from .networks import NN 
from scipy.stats import norm
import numpy as np
from .power_spectra import compute_smoothed_power_spectra
from .linalg_solve import mps_linalg_solve
import pandas as pd 
from scipy.optimize import linear_sum_assignment
from .tendencies import jacobian
import xarray as xr
from .ema import EMA 

import hashlib, random
import numpy as np, torch

def derived_seed(base_seed: int, tag: str) -> int:
    h = hashlib.sha1(f"{base_seed}:{tag}".encode()).hexdigest()
    return int(h[:8], 16)  # 32-bit

def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class KGAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim, 
        is_variational = False, 
        hidden_layers = [248, 248],
        activation = nn.Tanh, 
        power_spectrum_smoothing_kernel = 7,
        device = 'mps',
        fit_power = 4,
        seed = None,
        tag = ""
    ):
        super(KGAE, self).__init__()

        seed if seed is not None else np.random.randint(1000)
        self.seed = derived_seed(seed, tag)
        print(f"RANDOM SEED IS {self.seed}")
        seed_all(self.seed)

        self.input_dim = input_dim 
        self.latent_dim = latent_dim
        self.is_variational = is_variational
        self.activation = activation 
        self.power_spectrum_smoothing_kernel = power_spectrum_smoothing_kernel
        self.fit_power = fit_power 

        self.encoder = NN(
            self.input_dim, 
            2 * self.latent_dim if self.is_variational else self.latent_dim,
            hidden_layers = hidden_layers,
            activation = self.activation
        )

        self.decoder = NN(
            self.latent_dim, 
            self.input_dim,
            hidden_layers = hidden_layers[::-1],
            activation = self.activation
        )
        self.flip_signs = xr.DataArray(np.ones(self.latent_dim), coords={'mode': np.arange(1, self.latent_dim+1)}, dims=['mode'])

        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.to(self.device)

    def compute_salient_variance(self, x):
        if self.is_variational:
            # encode with variational encoder 
            mu_lv = self.encoder(x)
            mu_lv.retain_grad()
            mu, lv = mu_lv[:, :self.latent_dim], mu_lv[:, self.latent_dim:]
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu).to(mu.device)
        else:
            # encode with deterministic encoder
            z = self.encoder(x)
        z.retain_grad()

        decoded = self.decoder(z)
        decoded.backward(torch.ones_like(decoded).to(torch.float32).to(self.device), retain_graph=True) 
        saliency_map = torch.abs(z.grad) 
        saliency_map = saliency_map / saliency_map.sum(dim=1).reshape(-1,1)
        squared_residuals = (decoded - decoded.mean(dim=0).reshape(1,-1))**2 

        variance_fractions = []
        for i in range(z.shape[1]):
            variance_fractions.append(  (squared_residuals * saliency_map[:,i].reshape(-1,1)).mean(dim=0).sum() )
        variance_fractions = torch.hstack(variance_fractions)
        return variance_fractions / variance_fractions.sum() 

    def compute_loading_correlations(self, val=1.0):
        eye = torch.eye(self.latent_dim).to(self.device) * val
        dec1 = self.decoder(eye)
        corr =  torch.abs(torch.corrcoef(dec1 - dec1.mean(dim=1).reshape(-1,1)))
        offdiag = corr - torch.eye(self.latent_dim).to(self.device) * corr +1e-8 
        return offdiag

    def compute_loss(self, x, salient_variances):
        loss_terms = {}

        if self.is_variational:
            # encode with variational encoder 
            mu_lv = self.encoder(x)
            mu, lv = mu_lv[:, :self.latent_dim], mu_lv[:, self.latent_dim:]
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu).to(mu.device)

            # compute Kulback-Liebler Divergence Loss function (with N(0, 1) as prior)
            kl_divergence =  ( -0.5* (1 + lv - mu**2 - torch.exp(lv) ) ).mean() 
            loss_terms['KL Divergence'] = kl_divergence

        else:
            # encode with deterministic encoder
            z = self.encoder(x)
            mu = z 

        if np.isnan(np.min(z.cpu().detach().numpy())):
            print("NAN DETECTED IN LATENT VARIABLES")
            print(z)
            import sys; sys.exit()
        

        # decode
        x_hat = self.decoder(z)

        # Compute Reconstruction loss -log p(x|z) = ||X-Xhat||_2^2
        reconstruction_mse = torch.nn.functional.mse_loss(x, x_hat).mean()
        loss_terms['Reconstruction MSE'] = reconstruction_mse

        # Compute Join Spectral Overlap / Spatial Correlation Loss 
        power_spectrum, freqs = compute_smoothed_power_spectra(mu, kernel_size=self.power_spectrum_smoothing_kernel, dx=5)

        weights = torch.sqrt(torch.matmul(salient_variances.reshape(-1,1), salient_variances.reshape(1,-1)))
        weights = weights * (1 - torch.eye(weights.shape[0]).to(self.device))
        weights = weights / weights.sum() 

        ious, unis = 0*weights, 0*weights
        for indx in range(self.latent_dim):
            cur = power_spectrum[:,indx].reshape(-1,1)
            intersections = torch.hstack([ torch.minimum(cur, power_spectrum[:, inin].reshape(-1,1)) for inin in range(self.latent_dim) ])
            intersection_area = torch.sum((intersections[:-1,:] + intersections[1:,:]) / 2, dim=0)
            unions = torch.hstack([ torch.maximum(cur, power_spectrum[:, inin].reshape(-1,1)) for inin in range(self.latent_dim) ])
            union_area = torch.sum((unions[:-1,:] + unions[1:,:]) / 2, dim=0)
            total_iou = intersection_area / union_area 
            ious[indx, :] = intersection_area / union_area
            unis[indx, :] = union_area
        
        spatial_correlations = self.compute_loading_correlations(val=1)
        loss_terms['Spectral Overlap'] = torch.sum(weights*ious)
        loss_terms['Spatial Correlations'] = torch.sum(weights*spatial_correlations).detach()
        loss_terms['Spectral Weighted Spatial Correlations'] = torch.sum(weights * ious * spatial_correlations) 


        # compute curve-fit loss 
        B = torch.cumsum(power_spectrum, dim=0) * (1 - 1e-5)
        X = torch.hstack([ torch.linspace(-5,5, freqs.shape[0]).to(self.device).reshape(-1,1)**ji for ji in range(self.fit_power) ])
        if self.device == torch.device('cpu'):
          XTX = torch.matmul(X.t(), X)
          XTX_inv = torch.linalg.pinv(XTX ) 
          XtB = torch.matmul( X.t(), torch.log( B / (1-B)) )
          beta = torch.matmul( XTX_inv,  XtB)
        else: 
          Blogs = torch.log( B / (1-B))
          beta = mps_linalg_solve(X, Blogs)

        fit = torch.matmul(X, beta)
        logsig = F.sigmoid(fit)
        recreated = logsig * (1- logsig)
        recreated = recreated / recreated.sum(dim=0)
        mae = torch.abs(power_spectrum - recreated)
        loss_terms['Curve Fit MAE'] = torch.mean(mae.sum(dim=0))
        loss_terms['L1 Decoder Reg'] = torch.abs(1 - torch.norm(self.decoder.network[0].weight, p=2))
        loss_terms['Centering Loss'] = torch.abs(mu.mean(dim=0)).mean() 
        return loss_terms

    def fit(self, 
        training_data, 
        val_data, 
        num_epochs=500, 
        lr = 1e-3, 
        loss_weights = {
            'KL Divergence': 1e-2, 
            'Reconstruction MSE': 1, 
            'Spectral Overlap': 1,
            'Spatial Correlations': 0,
            'Spectral Weighted Spatial Correlations': 1,
            'Curve Fit MAE': 1,
            'L1 Decoder Reg': 1e-4, 
            'Centering Loss': 1
        },
        losses_to_use = [
            'KL Divergence', 
            'Reconstruction MSE', 
            'Spectral Overlap',
            'Spectral Weighted Spatial Correlations',
            'Curve Fit MAE',
            'L1 Decoder Reg', 
            'Centering Loss'
        ]
    ):
        train_loader = [ torch.tensor(minibatch, dtype=torch.float32).to(self.device) for minibatch in training_data ]
        val_loader = [ torch.tensor(val_data, dtype=torch.float32).to(self.device) ]

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        loss_tracking = {'train': {}, 'val': {} }

        ema = EMA(self, decay=0.995)
        minibatch_ndxs = np.arange(len(train_loader))
        for epoch in range(num_epochs):
            start = pd.Timestamp.now() 

            self.train()
            train_losses, val_losses = {}, {}
            np.random.shuffle(minibatch_ndxs)

            total_samples = 0
            for minibatch in minibatch_ndxs:
                self.optimizer.zero_grad() 

                samples_in_minibatch = train_loader[minibatch].shape[0] 
                total_samples += samples_in_minibatch 

                # compute weights corresponding to each pair of latent dims, based on their salient variances
                # with comparisons where i == j set to zero, all of which normalized to one. 
                salient_variances = self.compute_salient_variance(train_loader[minibatch])
                self.optimizer.zero_grad()

                losses = self.compute_loss(train_loader[minibatch], salient_variances)

                for key in losses.keys():
                    if key not in train_losses.keys():
                        train_losses[key] = samples_in_minibatch * losses[key]
                    else:
                        train_losses[key] += samples_in_minibatch * losses[key]
                
                loss_to_backprop = 0
                for key in loss_weights.keys(): 
                    if key in losses.keys() and key in losses_to_use:
                        loss_to_backprop += loss_weights[key] * losses[key]

                loss_to_backprop.backward() 
                self.optimizer.step() 
                ema.update(self)
            
            for key in train_losses.keys():
                train_losses[key] = train_losses[key] / total_samples
            
            self.eval() 
            ema.apply_shadow(self)

            self.zero_grad()
            salient_variances = self.compute_salient_variance(val_loader[0])
            self.optimizer.zero_grad()
            losses = self.compute_loss(val_loader[0], salient_variances)
            for key in losses.keys():
                if key not in val_losses.keys():
                    val_losses[key] = val_loader[0].shape[0] * losses[key]
                else:
                    val_losses[key] += val_loader[0].shape[0] * losses[key]

            for key in val_losses.keys():
                val_losses[key] = val_losses[key] / val_loader[0].shape[0]
            
            end = pd.Timestamp.now()
            print(f"---------------- EPOCH: {epoch} ------------------------")
            print("                 Train         Val")
            for key in val_losses.keys():
                if key in list(train_losses.keys()):
                    print(f"  {key}:  {train_losses[key]:>0.7f}  {val_losses[key]:>0.7f}") 
                    if key not in loss_tracking['train'].keys():
                        loss_tracking['train'][key] = []
                    loss_tracking['train'][key].append( train_losses[key].item() )

                else:
                    print(f"    {key}:  {val_losses[key]:>0.7f}") 

                if key not in loss_tracking['val'].keys():
                    loss_tracking['val'][key] = []
                loss_tracking['val'][key].append( val_losses[key].item() )
            print('Elapsed: ', end - start)
            print()
            ema.restore(self)

        ema.apply_shadow(self)
        return loss_tracking 
    

    def fit_noval(self, 
        training_data, 
        num_epochs=500, 
        lr = 1e-3, 
        loss_weights = {
            'KL Divergence': 1e-2, 
            'Reconstruction MSE': 1, 
            'Spectral Overlap': 1,
            'Spatial Correlations': 0,
            'Spectral Weighted Spatial Correlations': 1,
            'Curve Fit MAE': 1,
            'L1 Decoder Reg': 1e-4, 
            'Centering Loss': 1
        },
        losses_to_use = [
            'KL Divergence', 
            'Reconstruction MSE', 
            'Spectral Overlap',
            'Spectral Weighted Spatial Correlations',
            'Curve Fit MAE',
            'L1 Decoder Reg', 
            'Centering Loss'
        ]
    ):
        train_loader = [ torch.tensor(minibatch, dtype=torch.float32).to(self.device) for minibatch in training_data ]

        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        loss_tracking = {'train': {}, 'val': {} }

        ema = EMA(self, decay=0.995)
        minibatch_ndxs = np.arange(len(train_loader))
        for epoch in range(num_epochs):
            start = pd.Timestamp.now() 

            self.train()
            train_losses, val_losses = {}, {}
            np.random.shuffle(minibatch_ndxs)

            total_samples = 0
            for minibatch in minibatch_ndxs:
                self.optimizer.zero_grad() 

                samples_in_minibatch = train_loader[minibatch].shape[0] 
                total_samples += samples_in_minibatch 

                # compute weights corresponding to each pair of latent dims, based on their salient variances
                # with comparisons where i == j set to zero, all of which normalized to one. 
                salient_variances = self.compute_salient_variance(train_loader[minibatch])
                self.optimizer.zero_grad()

                losses = self.compute_loss(train_loader[minibatch], salient_variances)

                for key in losses.keys():
                    if key not in train_losses.keys():
                        train_losses[key] = samples_in_minibatch * losses[key]
                    else:
                        train_losses[key] += samples_in_minibatch * losses[key]
                
                loss_to_backprop = 0
                for key in loss_weights.keys(): 
                    if key in losses.keys() and key in losses_to_use:
                        loss_to_backprop += loss_weights[key] * losses[key]

                loss_to_backprop.backward() 
                self.optimizer.step() 
                ema.update(self)
            
            for key in train_losses.keys():
                train_losses[key] = train_losses[key] / total_samples
            
            end = pd.Timestamp.now()
            print(f"---------------- EPOCH: {epoch} ------------------------")
            print("                 Train         ")
            for key in train_losses.keys():
                print(f"  {key}:  {train_losses[key]:>0.7f}") 
                if key not in loss_tracking['train'].keys():
                    loss_tracking['train'][key] = []
                loss_tracking['train'][key].append( train_losses[key].item() )

            print('Elapsed: ', end - start)
            print()

        ema.apply_shadow(self)
        return loss_tracking 

    def align_latent_dims(self, reference_model, data, method='jacobian', dx=5):
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32)
            if method == 'jacobian':
                self_tendencies = jacobian(self, data.to(self.device), device=self.device)
                self_tendencies = self_tendencies.mean(dim=0).cpu().detach().numpy().T

                reference_tendencies = jacobian(reference_model, data.to(reference_model.device), device=reference_model.device)
                reference_tendencies = reference_tendencies.mean(dim=0).cpu().detach().numpy().T
            elif method == 'psd':
                # compute power spectra of latent variables for both models 
                with torch.no_grad():
                    z_self = self.encoder(data.to(self.device))
                    z_ref = reference_model.encoder(data.to(reference_model.device))

                psd_self, _ = compute_smoothed_power_spectra(z_self, kernel_size=self.power_spectrum_smoothing_kernel, dx=dx)
                psd_ref, _ = compute_smoothed_power_spectra(z_ref, kernel_size=reference_model.power_spectrum_smoothing_kernel, dx=dx)

                self_tendencies = psd_self.cpu().detach().numpy()
                reference_tendencies = psd_ref.cpu().detach().numpy() 
            else:
                raise ValueError("Method must be 'tendency' or 'psd'")         

            correlation_matrix = np.corrcoef(reference_tendencies.T, self_tendencies.T)[:self.latent_dim, self.latent_dim:]
            print("Correlation matrix between reference and current model latents:")
            print(correlation_matrix)
            
            # Find the best matching latent dimensions
            row_ind, col_ind = linear_sum_assignment(-np.abs(correlation_matrix))
            decoder_col_ind = col_ind.copy()
            if self.is_variational:
                col_ind = np.concatenate([col_ind, col_ind + self.latent_dim])
            print(col_ind)

            # Reorder the weights of the encoder and decoder networks
            self.encoder.network[-1].weight.data = self.encoder.network[-1].weight.data[col_ind, :]
            self.encoder.network[-1].bias.data   = self.encoder.network[-1].bias.data[col_ind]  # ADD THIS

            self.decoder.network[0].weight.data = self.decoder.network[0].weight.data[:, decoder_col_ind]

            # determine if we need to flip any dimensions
            sign = np.sign(correlation_matrix[np.arange(len(decoder_col_ind)), decoder_col_ind])
            sign[sign == 0] = 1
            self.flip_signs = xr.DataArray(sign, coords={'mode': np.arange(1, self.latent_dim+1)}, dims=['mode'])

        print("Aligned latent dimensions to reference model.")


    def sort_latents_by_frequency(self, data, dx=5, smoothing_kernel=15, reference_pattern=None):
        self.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32)
            z = self.encoder(data).detach().clone().requires_grad_(True)  # [B, L]
            if self.is_variational:
                mu, lvar = torch.chunk(z, 2, dim=1)
                z = mu  # use mean only for tendencies

            power_spectrum, freqs = compute_smoothed_power_spectra(z, kernel_size=smoothing_kernel, dx=dx)
            peak_ndxs = power_spectrum.argmax(dim=0)
            frequencies_of_max_power = freqs[peak_ndxs, :].diag() 
            col_ind = torch.argsort(frequencies_of_max_power).flip(0).cpu().detach().numpy() # sort highest-freq to lowest-freq

            decoder_col_ind = col_ind.copy()
            if self.is_variational:
                col_ind = np.concatenate([col_ind, col_ind + self.latent_dim])

            # Reorder the weights of the encoder and decoder networks
            self.encoder.network[-1].weight.data = self.encoder.network[-1].weight.data[col_ind, :]
            self.encoder.network[-1].bias.data   = self.encoder.network[-1].bias.data[col_ind]  # ADD THIS

            self.decoder.network[0].weight.data = self.decoder.network[0].weight.data[:, decoder_col_ind]

            # now we need to align signs to reference timeseries 
            self_tendencies = jacobian(self, data.to(self.device), device=self.device)
            self_tendencies = self_tendencies.mean(dim=0).cpu().detach().numpy().T
            
            correlation_matrix = np.corrcoef(self_tendencies.T, reference_pattern.reshape(1,-1))[:self.latent_dim, self.latent_dim:]

            sign = np.sign(correlation_matrix.squeeze())
            sign[sign == 0] = 1
            self.flip_signs = xr.DataArray(sign, coords={'mode': np.arange(1, self.latent_dim+1)}, dims=['mode'])
        
        print('Sorted model latents by frequency.')
        print("Aligned model to ENSO.")