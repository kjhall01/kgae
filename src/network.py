import torch 
import torch.nn as nn
import numpy as np
import xarray as xr 
from scipy import stats 
from torch.nn import functional as F 
import torch.nn.init as init 
from scipy.stats import norm 
from .utilities import calc_wnl
import sys
import matplotlib.pyplot as plt 

class PrintLayer(nn.Module):
  def __init__(self):
      super().__init__() 
  def forward(self, x):
      if 0:
          print(x.shape)
          print(x.mean())
          print(x.min(), x.max())
          print(torch.quantile( x, 0.05), torch.quantile( x, 0.95))
          print()
      return x 

class Network(nn.Module):
    def __init__(self, m=7,  iou_calc='joint', noise=None, fit_power=2, variance_calc='salient', scheme='mult', dwl_loss=True, bottleneck_dim=3, stout=None, lr=0.001, verbose=True, activation=nn.Sigmoid, hidden_layers=[], decoder_hidden_layers=None   ):
        super(Network, self).__init__()
        self.lr = lr
        self.scheme = scheme
        self.fit_power = fit_power
        self.m = m
        self.hidden_layers= hidden_layers
        self.decoder_hidden_layers = hidden_layers[::-1] if decoder_hidden_layers is None else decoder_hidden_layers
        self.activation=activation
        self.verbose=verbose
        self.bottleneck_dim = bottleneck_dim
        self.stout = stout if stout is not None else sys.stdout
        self.mul_factor = 'mse'
        self.variance_calc = variance_calc #_calc
        self.dwl_loss = dwl_loss
        self.iou_calc = iou_calc

    def get_frequency_weighted_var_and_means(self, data,  dx=5):
        N, M = data.shape
        average_dx = dx            
        data = data - data.mean(dim=0)

        frequencies = torch.fft.rfftfreq(data.shape[0], d=average_dx)
        fourier_coeffs = torch.fft.rfft(data, dim=0)
        fourier_coeffs = (fourier_coeffs*torch.conj(fourier_coeffs)).real
        fc_scale = fourier_coeffs.sum(dim=0) 
        fourier_coeffs = fourier_coeffs / fc_scale

        m_for_deniell_smoothing = self.m 
        if m_for_deniell_smoothing > 0:
          padded = F.pad(fourier_coeffs.t(), (m_for_deniell_smoothing//2 , m_for_deniell_smoothing//2), mode='reflect').t()
          freqs = torch.vstack([frequencies.reshape(1,-1), frequencies.reshape(1,-1)]) 
          padded_freqs = F.pad(freqs, (m_for_deniell_smoothing//2, m_for_deniell_smoothing//2), mode='reflect')
          tc = [] 
          tc2 = []
          smoothing_weights = norm.pdf(np.linspace(-2,2, m_for_deniell_smoothing))
          smoothing_weights = smoothing_weights / smoothing_weights.sum()
          for i in range(m_for_deniell_smoothing):
              tcc =  padded[i: i+fourier_coeffs.shape[0], : ]
              tcc2 = padded_freqs[0, i:i+fourier_coeffs.shape[0]].squeeze()
              tc.append(tcc *smoothing_weights[i] )
              tc2.append(tcc2 *smoothing_weights[i] )

          fourier_coeffs = torch.dstack(tc).sum(dim=-1)
          frequencies = torch.dstack(tc2).sum(dim=-1).squeeze()
        sums = fourier_coeffs.sum(dim=0)
        fourier_coeffs = fourier_coeffs / sums
        #frequencies[frequencies ==0 ] = 1e-8
        #frequencies = 1 / frequencies
        xbar_standard = 0.5*frequencies.max() + 0.5*frequencies.min()
        max_spread = torch.sqrt(0.5*(frequencies.max() -xbar_standard)**2 + 0.5*(frequencies.min() - xbar_standard)**2 ) #torch.mean((frequencies.reshape(-1,1)*fourier_coeffs - xbar_standard)**2)

        frequency_weighted_mean = torch.sum(frequencies.reshape(-1,1)* fourier_coeffs, dim=0)
        peak_ndcs = torch.argmax(fourier_coeffs, dim=0)

        spectral_peaks = 1/frequencies[peak_ndcs]       
        frequency_weighted_var =torch.sqrt(torch.sum(fourier_coeffs*(frequencies.reshape(-1,1) -frequency_weighted_mean)**2 , dim=0) )

        return spectral_peaks / 12, 1/frequency_weighted_mean / 12, frequency_weighted_var

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def transform(self, x, keep=None):
        keep = self.bottleneck_dim if keep is None else keep        
        data_np = x.astype(np.float32)
        data_tensor = torch.tensor(data_np, dtype=torch.float32)

        with torch.no_grad():
            encoded = self.encoder(data_tensor)
            for i in range(self.bottleneck_dim):
                if i >= keep:
                    encoded[:, self.sort_order[i]] = encoded.mean(dim=0)[self.sort_order[i]]
            test_output = self.decoder(encoded)
        return test_output.detach().numpy()

    def compute_variance_fracs(self, data_tensor):
      true_data_var = data_tensor.var(dim=0).sum() 
      encoded1 = self.encoder(data_tensor)
      dec1 = self.decoder(encoded1)
      orig_decoded_var = dec1.var(dim=0).sum()

      pcts_lost = []
      for i in range(self.bottleneck_dim):
        encoded = encoded1.clone()
        encoded[:, i] = encoded.mean(dim=0)[i]
        dec = self.decoder(encoded)
        modified_decoded_var = dec.var(dim=0).sum() 
        pcts_lost.append(torch.abs(1 - modified_decoded_var / orig_decoded_var))
      pcts_lost = torch.hstack(pcts_lost)
      pcts_lost = pcts_lost / pcts_lost.sum()
      return pcts_lost, pcts_lost * orig_decoded_var / true_data_var

    def curve_fit_loss(self, data, var_fracs, dx=5):
        N, M = data.shape
        average_dx = dx    
        frequencies = torch.fft.rfftfreq(data.shape[0], d=average_dx)
    
        data = data - data.mean(dim=0)

        frequencies = torch.fft.rfftfreq(data.shape[0], d=average_dx)
        fourier_coeffs = torch.fft.rfft(data, dim=0)
        fourier_coeffs = (fourier_coeffs*torch.conj(fourier_coeffs)).real
        fc_scale = fourier_coeffs.sum(dim=0) 
        fourier_coeffs = fourier_coeffs / fc_scale

        m_for_deniell_smoothing = self.m 
        if m_for_deniell_smoothing > 0:
          padded = F.pad(fourier_coeffs.t(), (m_for_deniell_smoothing//2 , m_for_deniell_smoothing//2), mode='reflect').t()
          freqs = torch.vstack([frequencies.reshape(1,-1), frequencies.reshape(1,-1)]) 
          padded_freqs = F.pad(freqs, (m_for_deniell_smoothing//2, m_for_deniell_smoothing//2), mode='reflect')
          tc = [] 
          tc2 = []
          smoothing_weights = norm.pdf(np.linspace(-2,2, m_for_deniell_smoothing))
          smoothing_weights = smoothing_weights / smoothing_weights.sum()
          for i in range(m_for_deniell_smoothing):
              tcc =  padded[i: i+fourier_coeffs.shape[0], : ]
              tcc2 = padded_freqs[0, i:i+fourier_coeffs.shape[0]].squeeze()
              tc.append(tcc *smoothing_weights[i] )
              tc2.append(tcc2 *smoothing_weights[i] )

          fourier_coeffs = torch.dstack(tc).sum(dim=-1)
          frequencies = torch.dstack(tc2).sum(dim=-1).squeeze()
        sums = fourier_coeffs.sum(dim=0)
        fourier_coeffs = fourier_coeffs / sums

        B = torch.cumsum(fourier_coeffs, dim=0) * (1 - 1e-5)
        #print(B.shape, B.min(), B.max())
        X = torch.hstack([ torch.linspace(-5,5, frequencies.shape[0]).reshape(-1,1)**ji for ji in range(self.fit_power) ])
       # print(X.shape)
        XTX = torch.matmul(X.t(), X)
        pinv = torch.linalg.pinv(XTX )
        beta = torch.matmul( pinv,  torch.matmul( X.t(),  torch.log( B / (1-B))))
        fit = torch.matmul(X, beta)
        logsig = F.sigmoid(fit)
        recreated = logsig * (1- logsig)
        recreated = recreated / recreated.sum(dim=0)
        mse = torch.abs(fourier_coeffs - recreated)
        return torch.mean(mse.sum(dim=0))


    def compute_loading_correlations(self, val=1.0):
      encoded1 = torch.eye(self.bottleneck_dim).to(torch.float32) * val
      dec1 = self.decoder(encoded1)
      corr =  torch.abs(torch.corrcoef(dec1 - dec1.mean(dim=1).reshape(-1,1)))
      offdiag = corr - torch.eye(self.bottleneck_dim) * corr +1e-8 
      return offdiag

    def encode(self, x): 
        data_np = x.astype(np.float32)
        data_tensor = torch.tensor(data_np, dtype=torch.float32)

        with torch.no_grad():
            encoded, test_output = self(data_tensor)
        return encoded.detach().numpy()

    def correlation_penalty(self, encoded, var_fracs):
        encoded = encoded - encoded.mean(dim=0)
        covariance = torch.cov(encoded.T) 
        vf_rows = torch.ones_like(covariance) * var_fracs 
        vf_cols = torch.ones_like(covariance) * var_fracs.reshape(-1,1)

        weights = vf_rows * vf_cols
        corr = torch.corrcoef(encoded.T)
        offdiag = corr - torch.eye(encoded.shape[1]) * corr +1e-8 
        penalty = torch.sum(torch.abs(offdiag) * weights) / (weights - torch.eye(encoded.shape[1])*weights).sum()
        return penalty
    
    def spectral_overlap_loss(self, data, var_fracs=None, dx=5):
        N, M = data.shape
        average_dx = dx    
        frequencies = torch.fft.rfftfreq(data.shape[0], d=average_dx)
    
        data = data - data.mean(dim=0)

        frequencies = torch.fft.rfftfreq(data.shape[0], d=average_dx)
        fourier_coeffs = torch.fft.rfft(data, dim=0)
        fourier_coeffs = (fourier_coeffs*torch.conj(fourier_coeffs)).real
        fc_scale = fourier_coeffs.sum(dim=0) 
        fourier_coeffs = fourier_coeffs / fc_scale

        
        m_for_deniell_smoothing = self.m 
        if m_for_deniell_smoothing > 0:
          padded = F.pad(fourier_coeffs.t(), (m_for_deniell_smoothing//2 , m_for_deniell_smoothing//2), mode='reflect').t()
          freqs = torch.vstack([frequencies.reshape(1,-1), frequencies.reshape(1,-1)]) 
          padded_freqs = F.pad(freqs, (m_for_deniell_smoothing//2, m_for_deniell_smoothing//2), mode='reflect')
          tc = [] 
          tc2 = []
          smoothing_weights = norm.pdf(np.linspace(-2,2, m_for_deniell_smoothing))
          smoothing_weights = smoothing_weights / smoothing_weights.sum()
          for i in range(m_for_deniell_smoothing):
              tcc =  padded[i: i+fourier_coeffs.shape[0], : ]
              tcc2 = padded_freqs[0, i:i+fourier_coeffs.shape[0]].squeeze()
              tc.append(tcc *smoothing_weights[i] )
              tc2.append(tcc2 *smoothing_weights[i] )

          fourier_coeffs = torch.dstack(tc).sum(dim=-1)
          frequencies = torch.dstack(tc2).sum(dim=-1).squeeze()
        sums = fourier_coeffs.sum(dim=0)
        fourier_coeffs = fourier_coeffs / sums
    
        pcts = torch.matmul(var_fracs.reshape(-1,1), var_fracs.reshape(1,-1))
        vf_rows = torch.ones_like(pcts) * torch.sqrt(var_fracs)
        vf_cols = torch.ones_like(pcts) * torch.sqrt(var_fracs.reshape(-1,1))

        if self.scheme == 'mult':
            pcts = vf_rows * vf_cols
        else:
           pcts = vf_rows + vf_cols
        pcts = pcts - torch.eye(var_fracs.shape[0])*pcts 
        pcts = pcts / pcts.sum() 

        pcts1 = self.compute_loading_correlations(val=1)
        pcts2 = self.compute_loading_correlations(val=-1)

        ious = 0*pcts2 
        unis = 0*pcts2
        if self.iou_calc == 'separate':
          dx = 1
          for indx in range(self.bottleneck_dim):
            cur = fourier_coeffs[:,indx].reshape(-1,1)
            intersections = torch.hstack([ torch.minimum(cur, fourier_coeffs[:, inin].reshape(-1,1)) for inin in range(self.bottleneck_dim) ])
            intersection_area = torch.sum((intersections[:-1,:] + intersections[1:,:])*dx / 2, dim=0)
            unions = torch.hstack([ torch.maximum(cur, fourier_coeffs[:, inin].reshape(-1,1)) for inin in range(self.bottleneck_dim) ])
            union_area = torch.sum((unions[:-1,:] + unions[1:,:])*dx / 2, dim=0)
            total_iou = intersection_area / union_area 
            ious[indx, :] = intersection_area / union_area
            unis[indx, :] = union_area
        else:
          assert False, 'no option for joint iou anymore'
        return  torch.sum(pcts*ious)  + torch.sum(pcts*ious*pcts1)


    def fit(self, training_data=None, val_data=None, val_dx=1, dx=5, num_epochs=3000, patience=5, alpha=0.05, outpath='best_model.pth' ):

        self.input_dim = training_data[0].shape[1]
        self.hidden_layers.insert(0, self.input_dim)        
        self.decoder_hidden_layers.insert(0, self.bottleneck_dim)

        # construct encoder 
        if len(self.hidden_layers) == 0:
            self.encoder = nn.Sequential(*[  nn.Linear(self.input_dim, self.bottleneck_dim)])
        else:
            enc = []
            while len(self.hidden_layers) > 1: 
                laya = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])
                init.xavier_normal_(laya.weight)
                init.zeros_(laya.bias)
                #print(torch.norm(laya.weight, p=2))
               # print(torch.norm(laya.bias, p=2))
                enc.append(laya)
                enc.append(PrintLayer())

                enc.append(self.activation())
                self.hidden_layers.pop(0)
            laya = nn.Linear( self.hidden_layers[0], self.bottleneck_dim)
            #init.xavier_normal_(laya.weight)
            #init.zeros_(laya.bias)
            enc.append(laya)
            enc.append(PrintLayer())
            #print(torch.norm(laya.weight, p=2))
          #  print(torch.norm(laya.bias, dim=0, p=2))

            self.encoder = nn.Sequential(*enc)


        # construct decoder 
        if len(self.decoder_hidden_layers) == 0:
            self.decoder = nn.Sequential(nn.Linear(self.bottleneck_dim, self.input_dim))
        else:
            dec = []
            while len(self.decoder_hidden_layers) > 1: 
                laya = nn.Linear(self.decoder_hidden_layers[0], self.decoder_hidden_layers[1])
                init.xavier_normal_(laya.weight)
                init.zeros_(laya.bias)
                dec.append(laya)
                dec.append(self.activation())
                self.decoder_hidden_layers.pop(0)
            dec.append(nn.Linear( self.decoder_hidden_layers[0], self.input_dim))
            self.decoder = nn.Sequential(*dec)

        train_loader = [ torch.tensor(td, dtype=torch.float32) for iii, td in enumerate(training_data)  ]
        val_loader = [ torch.tensor(val_data, dtype=torch.float32) ]
        tracking, val_tracking = [], []
        mse_tracking, val_mse_tracking = [], [] 
        fft_tracking, val_fft_tracking = [], [] 
        spread_tracking, val_spread_tracking = [], [] 

        self.criterion = nn.MSELoss()  # Mean Squared Error Loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.crit2 = nn.MSELoss() 

        best_loss = float('inf')

        epoch = 0
        while epoch < num_epochs and self.lr < 1:
            epoch += 1
            self.epoch = epoch
            self.train()
            shf_ncds = np.asarray([ train_ndx for train_ndx in range(len(training_data))  ])
            np.random.shuffle(shf_ncds)
            running_loss = [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]]
          
            for batch in [train_loader[ndxx] for ndxx in shf_ncds]:
              self.optimizer.zero_grad()
              total_loss = 0
              nreps = 1#2 if epoch > patience  else 1 
              for indx1 in range(nreps):
                if indx1 > 0:
                   indx = 1
                else:
                   indx = 0
                encoded = self.encoder(batch)

                encoded.retain_grad()
                if self.variance_calc == 'lost':
                  var_frac_pcts, variance_fractions  = self.compute_variance_fracs(batch)
                else: 
                  true_variance = batch.var(dim=0).sum()
                  decoded = self.decoder(encoded)#.detach()
                  decoded.backward(torch.ones_like(decoded).to(torch.float32), retain_graph=True) 
                  saliency_map = torch.abs(encoded.grad) 
                  saliency_map = saliency_map / saliency_map.sum(dim=1).reshape(-1,1)
                  squared_residuals = (decoded - decoded.mean(dim=0).reshape(1,-1))**2 

                  variance_fractions = []
                  for i in range(encoded.shape[1]):
                    variance_fractions.append(  (squared_residuals * saliency_map[:,i].reshape(-1,1)).mean(dim=0).sum() )
                  variance_fractions = torch.hstack(variance_fractions)
                  variance_fractions = variance_fractions / true_variance
                  var_frac_pcts = variance_fractions / variance_fractions.sum() 

                decoded = self.decoder(encoded)#.detach()


                self.optimizer.zero_grad()
                mse_loss = self.criterion(decoded, batch) 
                joint_iou_corr_loss = self.spectral_overlap_loss(encoded, var_fracs=var_frac_pcts, dx=dx)
                ps_spread = self.curve_fit_loss(encoded, var_frac_pcts, dx=dx)
                decoder_weight_loss = torch.abs(1 - torch.norm(self.decoder[0].weight, p=2))
                centered_loss =  torch.abs(encoded.mean(dim=0)).mean() 
                total_loss += mse_loss 
                total_loss += centered_loss  
                total_loss += ps_spread 
                total_loss += joint_iou_corr_loss

                if self.dwl_loss: 
                  total_loss += decoder_weight_loss

                losses = [total_loss, mse_loss, centered_loss, decoder_weight_loss, ps_spread,joint_iou_corr_loss ]# ps_spread, lag_corr_loss]
                for l in range(6):
                  running_loss[indx][l] += losses[l].item() 
              total_loss.backward()
              self.optimizer.step()


            for l in range(6):
               for l2 in range(2):
                 running_loss[l2][l] = running_loss[l2][l] / len(train_loader)
            tracking.append(running_loss[0][0])
            mse_tracking.append(running_loss[0][1])
            fft_tracking.append(running_loss[0][5])
            spread_tracking.append(running_loss[0][4])


            # validation
            val_running_loss = [0, 0, 0, 0, 0, 0]
            for batch in val_loader:
              self.eval()
              self.optimizer.zero_grad()
              encoded = self.encoder(batch)
              encoded.retain_grad()
              if self.variance_calc == 'lost':
                var_frac_pcts, variance_fractions  = self.compute_variance_fracs(batch)
              else: 
                true_variance = batch.var(dim=0).sum()
                decoded = self.decoder(encoded)#.detach()
                decoded.backward(torch.ones_like(decoded).to(torch.float32), retain_graph=True) 
                saliency_map = torch.abs(encoded.grad) 
                saliency_map = saliency_map / saliency_map.sum(dim=1).reshape(-1,1)
                squared_residuals = (decoded - decoded.mean(dim=0).reshape(1,-1))**2 

                variance_fractions = []
                for i in range(encoded.shape[1]):
                  variance_fractions.append(  (squared_residuals * saliency_map[:,i].reshape(-1,1)).mean(dim=0).sum() )
                variance_fractions = torch.hstack(variance_fractions)
                variance_fractions = variance_fractions / true_variance
                var_frac_pcts = variance_fractions / variance_fractions.sum() 
              decoded = self.decoder(encoded)#.detach()

              self.optimizer.zero_grad()
              mse_loss = self.criterion(decoded, batch) 

              joint_iou_corr_loss = self.spectral_overlap_loss(encoded,var_fracs=var_frac_pcts, dx=val_dx)
              ps_spread = self.curve_fit_loss(encoded, var_frac_pcts, dx=val_dx)
              centered_loss =  torch.abs(encoded.mean(dim=0)).mean() 
              decoder_weight_loss = torch.abs(1 - torch.norm(self.decoder[0].weight, p=2))
              
              total_loss = mse_loss + centered_loss  + ps_spread #+ lag_corr_loss #+ decoder_weight_loss
              total_loss += joint_iou_corr_loss
              if self.dwl_loss: #
                 total_loss = total_loss + decoder_weight_loss 

                 
              losses = [total_loss, mse_loss, centered_loss, decoder_weight_loss, ps_spread, joint_iou_corr_loss] #, ps_spread, lag_corr_loss]


              for l in range(6):
                val_running_loss[l] += losses[l].item() 

            stopping_metric = val_running_loss[1] 
            if stopping_metric < best_loss:
                torch.save(self.state_dict(), outpath)
                best_loss = stopping_metric
            val_tracking.append(val_running_loss[0] )
            val_mse_tracking.append(val_running_loss[1])
            val_fft_tracking.append(val_running_loss[5])
            val_spread_tracking.append(val_running_loss[4])


            if epoch < 2*patience +1:
              slope=-1
              p_value=0.5*alpha  
            else:
              means = []
              for ii in range(patience):
                index = len(tracking) - patience - patience + ii
                means.append(np.asarray(tracking)[index:index+patience].mean())
              xx = np.arange(patience)
              slope, intercept, r_value, p_value, std_err = stats.linregress(xx, np.asarray(means))

            if self.verbose:
              print(f'Epoch [{epoch + 1}/{num_epochs}] -- Total: {val_running_loss[0]:.2f}, MSE: {val_running_loss[1]:.2f}, Mean: {val_running_loss[2]:.2f}, DWL: {val_running_loss[3]:.2f},  FFT: {val_running_loss[5]:.2f}, PSSPD: {val_running_loss[4]:.2f}, SLOPE: {slope:.2f}, P: {p_value:.4f}', file=self.stout)
              print(f'         Train  -- Total: {running_loss[0][0]:.2f}, MSE: {running_loss[0][1]:.2f}, Mean: {running_loss[0][2]:.2f}, DWL: {running_loss[0][3]:.2f},  FFT: {running_loss[0][5]:.2f}, PSSPD: {running_loss[0][4]:.2f}, SLOPE: {slope:.2f}, P: {p_value:.4f}', file=self.stout)
              print(f'         Train(2)- Total: {running_loss[1][0]:.2f}, MSE: {running_loss[1][1]:.2f}, Mean: {running_loss[1][2]:.2f}, DWL: {running_loss[1][3]:.2f},  FFT: {running_loss[1][5]:.2f}, PSSPD: {running_loss[1][4]:.2f}, SLOPE: {slope:.2f}, P: {p_value:.4f}', file=self.stout)

            if p_value >= alpha:
                if self.verbose:
                  print('\nEarly stopping triggered.', file=self.stout)
                break
            
            if np.isnan(total_loss.item()):
                break
            
        self.eval()
        if self.verbose:
            print('\nTraining complete!', file=self.stout)

        self.load_state_dict(torch.load(outpath))

        # compute variance lost 
        data_np = val_data.astype(np.float32)
        data_tensor = torch.tensor(data_np, dtype=torch.float32)

        self.relative_variance_lost, self.absolute_variance_lost  = self.compute_variance_fracs(data_tensor)
        self.relative_variance_lost = self.relative_variance_lost.detach().numpy()
        self.absolute_variance_lost = self.absolute_variance_lost.detach().numpy() 

        # compute salient variance 
        self.optimizer.zero_grad()
        encoded = self.encoder(data_tensor)
        encoded.retain_grad()

        true_variance = data_tensor.var(dim=0).sum()
        decoded = self.decoder(encoded)#.detach()
        decoded.backward(torch.ones_like(decoded).to(torch.float32), retain_graph=True) 
        saliency_map = torch.abs(encoded.grad) 
        saliency_map = saliency_map / saliency_map.sum(dim=1).reshape(-1,1)
        squared_residuals = (decoded - decoded.mean(dim=0).reshape(1,-1))**2 

        variance_fractions = []
        for i in range(encoded.shape[1]):
          variance_fractions.append(  (squared_residuals * saliency_map[:,i].reshape(-1,1)).mean(dim=0).sum() )
        variance_fractions = torch.hstack(variance_fractions)
        variance_fractions = variance_fractions / true_variance
        var_frac_pcts = variance_fractions / variance_fractions.sum() 
 
        self.optimizer.zero_grad()
        
        self.relative_salient_variance = var_frac_pcts.detach().numpy()
        self.absolute_salient_variance = variance_fractions.detach().numpy() 
        
        self.spectral_peaks, self.frequency_weighted_means, self.frequency_variance = self.get_frequency_weighted_var_and_means(encoded, dx=val_dx)
        self.spectral_overlap = self.spectral_overlap_loss(encoded, var_fracs=var_frac_pcts, dx=val_dx)
        # compute loss function on validation set 
        mse_loss = self.criterion(decoded, data_tensor) * float(data_tensor.shape[0]) #* float(1 - epoch/num_epochs)

        self.sort_order = np.argsort(self.relative_variance_lost)[::-1]
       # self.sort_order = np.argsort(self.spectral_peaks)[::-1]
       # self.spectral_peaks = self.spectral_peaks.copy()[self.sort_order]
        self.spectral_peaks = np.asarray([self.spectral_peaks[self.sort_order[nm]] for nm in range(self.bottleneck_dim)])
        

        #else:
        #  self.sort_order = np.argsort(self.relative_salient_variance)[::-1]


        self.absolute_variance_lost = self.absolute_variance_lost[self.sort_order]
        self.relative_variance_lost = self.relative_variance_lost[self.sort_order]
        self.relative_salient_variance = self.relative_salient_variance[self.sort_order]
        self.absolute_salient_variance = self.absolute_salient_variance[self.sort_order]

       # self.frequency_weighted_means = fwm.detach().numpy()[self.sort_order]
        self.scores = torch.from_numpy(encoded.detach().numpy()[:, self.sort_order])
        self.correlations = np.corrcoef(self.scores.detach().numpy().T) 
        self.effective_bottleneck_nodes =  (np.arange(self.bottleneck_dim)+1).dot(self.relative_salient_variance.T) 
        
        # compute variance lost for TRAINING 
        data_tensor = torch.vstack(train_loader)#.astype(np.float32)

        self.relative_variance_lost_train, self.absolute_variance_lost_train  = self.compute_variance_fracs(data_tensor)
        self.relative_variance_lost_train = self.relative_variance_lost_train.detach().numpy()
        self.absolute_variance_lost_train = self.absolute_variance_lost_train.detach().numpy() 

        # compute salient variance 
        self.optimizer.zero_grad()
        encoded = self.encoder(data_tensor)
        encoded.retain_grad()

        true_variance = data_tensor.var(dim=0).sum()
        decoded = self.decoder(encoded)#.detach()
        decoded.backward(torch.ones_like(decoded).to(torch.float32), retain_graph=True) 
        saliency_map = torch.abs(encoded.grad) 
        saliency_map = saliency_map / saliency_map.sum(dim=1).reshape(-1,1)
        squared_residuals = (decoded - decoded.mean(dim=0).reshape(1,-1))**2 

        variance_fractions = []
        for i in range(encoded.shape[1]):
          variance_fractions.append(  (squared_residuals * saliency_map[:,i].reshape(-1,1)).mean(dim=0).sum() )
        variance_fractions = torch.hstack(variance_fractions)
        variance_fractions = variance_fractions / true_variance
        var_frac_pcts = variance_fractions / variance_fractions.sum() 
 
        self.optimizer.zero_grad()
        self.epochs = val_mse_tracking.index(best_loss)

        self.relative_salient_variance_train = var_frac_pcts.detach().numpy()
        self.absolute_salient_variance_train = variance_fractions.detach().numpy() 
        
        self.absolute_variance_lost_train = self.absolute_variance_lost_train[self.sort_order]
        self.relative_variance_lost_train = self.relative_variance_lost_train[self.sort_order]
        self.relative_salient_variance_train = self.relative_salient_variance_train[self.sort_order]
        self.absolute_salient_variance_train = self.absolute_salient_variance_train[self.sort_order]
        self.fwm = False
        return tracking, val_tracking, mse_tracking, val_mse_tracking, fft_tracking, val_fft_tracking, spread_tracking, val_spread_tracking


