import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from pathlib import Path 

srcdir = "."
srcdir = os.path.abspath(srcdir)
sys.path.insert(0,srcdir)
import src

# open ERA5 training data 
sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))

# we need an xr.DataArray full of ones, which is shaped like the original dataset to help with saving decoded data in netcdf
training_template = xr.ones_like(sst.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature'))

# split training data into 5-fold crossvalidation splits (taking every fifth value to be one group)
splits = [ sst.isel(time=slice(i, None, 5)) for i in range(5) ]

# open ERA5 test data 
test_sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(pd.Timestamp(2015,1,1), None))
test_sst = test_sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None)) #.sel(lat=slice(-13,13,None))

# we need an xr.DataArray full of ones, which is shaped like the original TEST dataset to help with saving decoded data in netcdf
test_template = xr.ones_like(test_sst.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature'))

# these are the names I am assigning to the latent dimensions, sorted by spectral peak
titles2 = ['Decadal', 'Interannual', 'Quasibiennial', 'HF1', 'HF2']

# this is the name of the top-top level output directory for this experiment 
run = 'basin.goodtest.recursive.crossvalidated.1940-2014'
N = 50  # number or random initializations 
M = 3 # number of recursions


# for random initialization in N trials:
for random_seed in range(N):

    # check if top-level output directory for this Random seed already exists - if so move on to next random initialization
    expdir_name = f'{run}/rs{random_seed}'
    if Path(expdir_name).is_dir():
        continue

    # set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


    # for crossvalidation fold in the five-folds:
    for split in range(5):

        # recombine the other four folds to calculate trend and climatology 
        train = xr.concat([splits[j] for j in range(5) if j != split ], 'time').sortby('time')
        train, fit, gwm, p  = src.global_detrend(train, deg=2)
        train, monthly_clim = src.remove_climo(train)

        # remove trend and climatology (that was calculated on training set) from validation data
        val = splits[split]
        val = val - xr.polyval(val['time'], p.polyfit_coefficients)

        # remove the climatology that was calculated on train- this is what is inside of src.remove_climo, but instead of calculating 
        # the climatology based on the data to be anomalized, we use the one from train
        dim, monthly ='time', val
        toconcat = []
        for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
            ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
            ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
            toconcat.append(ds_yearly)
        val = xr.concat(toconcat, dim).sortby(dim)

        # remove trend and climatology (that was calculated on training set) from test data
        test = test_sst - xr.polyval(test_sst['time'], p.polyfit_coefficients)

        # remove the climatology that was calculated on train- this is what is inside of src.remove_climo, but instead of calculating 
        # the climatology based on the data to be anomalized, we use the one from train
        dim, monthly ='time', test
        toconcat = []
        for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
            ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
            ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
            toconcat.append(ds_yearly)
        test = xr.concat(toconcat, dim).sortby(dim)


        # re-split the training data, taking every 4th value to be one split 
        # (because in train, it is 4/5 folds stacked together and sorted. so every fourth value belongs to the same split)
        # i.e., if  1 2 3 4 5 6 7 8 9 10 is the full training dataset (train + val, not test)
        # then the splits are [1, 6], [2, 7], [3, 8], [4, 9], [5, 10]
        # if the validation split is [2, 7], and the training splits are [1, 6], [3, 8], [4, 9], [5,10]
        # then train includes 1 3 4 5 6 8 9 10. the trend and climatology are calculated from that 
        # then we resplit by taking every fourth value in that: 
        # [1, 6], [3, 8], [4, 9], [5, 10]
        train_splits = [train.isel(time=slice(j, None, 5-1 )) for j in range(5-1)]

        # we also then stack the lat/lon dimension into a 'feature' dimension which will be the unique input features for the NN
        # that is done for each crossval fold with xarray, and then the numpy data is pulled out with .values 
        stacked = [ t.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') for t in train_splits ]
        training_data = [ t.values for t in stacked ]


        # extract numpy values of validation set 
        val = val.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') 
        val_template = xr.ones_like(val)
        val_data = val.values 

        # we already have test_template
        test_data = test.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature').values 

        # for each recursion (meaning, each time we train an autoencoder and remove the recreated data from the original data) 
        for recursion in range(M):

            # make the destination directory for this crossval split / recursion 
            out_dir = Path(expdir_name) / f'split{split}' / f'recursion{recursion}'
            out_dir.mkdir(exist_ok=True, parents=True)
            (out_dir / 'cesm').mkdir(exist_ok=True, parents=True)
            (out_dir / 'e3sm').mkdir(exist_ok=True, parents=True)

            # open NN training output file 
            stoutf = open(out_dir / 'terminal_output.txt', 'w') 

            # instantiate NN 
            oae = src.Network(
                lr=1e-3,
                activation=nn.Tanh,
                verbose=True,
                bottleneck_dim=5,
                hidden_layers=[248,248],
                noise=None,
                m=7,
                scheme='mult',
                fit_power=4,
                stout=stoutf,
                dwl_loss=True, 
                variance_calc='salient',
                iou_calc='separate',
            )

            # trainin NN 
            print(f"\n\n\n\n\n\n---------- TRAINING MODEL {split+1} --------------", file=stoutf)
            tracking, val_tracking, mse_tracking, val_mse_tracking, fft_tracking, val_fft_tracking, spread_tracking, val_spread_tracking = oae.fit(
                training_data=training_data,
                val_data=val_data,
                num_epochs=500,
                patience=30,
                outpath=  out_dir / f'model_parameters_split{split}.pth',
                dx=5,
                val_dx=5
            )

            # close NN training output file 
            stoutf.close()


            # reassemble validation and training INPUT data 
            training_data2 = [_ for _ in training_data]
            training_data2.insert(split, val_data)
            data = np.zeros((sum([td.shape[0] for td in training_data2]), training_data2[0].shape[1])).astype(float)
            data_indices = [[] for _ in range(len(training_data2))]
            for ii in range(data.shape[0]):
                data_indices[ii % 5].append(ii)
            for n, ndxs in enumerate(data_indices): 
                data[ndxs,:] = training_data2[n]   
            data123 = torch.tensor(data, dtype=torch.float32)

            # run it through Autoencoder
            enc_temp = oae.encoder(data123)
            decoded = oae.decoder(enc_temp)

            # extract encodings to numpy, and decodings to xarray 
            encoded = enc_temp.detach().numpy()
            decoded = (decoded.detach().numpy() * training_template).unstack('feature').sortby('lat').sortby('lon')
            
            # get the locations (in frequency space) of the spectral peaks of the encodings 
            peaks, fwm, fv = oae.get_frequency_weighted_var_and_means(enc_temp, dx=1)
            peaks = np.asarray([pk for pk in peaks])

            # the peak order will be the lowest frequency first, highest frequency last.
            # so decadal mode will be peak_order[0], interannual mode peak_order[1], etc 
            peak_order = np.argsort(peaks)[::-1]

            # since the Autoencoder might learn some arbitrary sign flipping,
            # we will flip the sign of each node based on its correlation with the eastern equatorial pacific.
            tropmean_sst = decoded.sel(lat=slice(-5,5), lon=slice(10, 60))
            tropmean_sst = tropmean_sst.mean('lat').mean('lon').values.reshape(-1,1)

            # decide whether each one should be flipped
            flips = [ False for _ in range(peak_order.shape[0])]
            for _, peak in enumerate(peak_order): 
                flips[_] = np.corrcoef(tropmean_sst.squeeze(), encoded[:, peak].squeeze())[-1,0] < 0

            # run JUST the validation data through the AE  
            val_data = torch.tensor(val_data, dtype=torch.float32)
            enc = oae.encoder(val_data)
            encoded = enc.detach().numpy()

            # flip the signs of the encodings according to correlation w/ trop pac
            for _, peak in enumerate(peak_order):
                if flips[_]:
                    encoded[:, peak] = encoded[:, peak] * -1 
            
            # save the validation encodings! and the validation decodings 
            timecoord =  [pd.Timestamp(i) for i in sst.time.isel(time=slice(split, None, 5)).values]
            da = xr.DataArray(data=encoded[:, peak_order], name='encodings', dims=('time', 'mode'), coords={'time': timecoord, 'mode':titles2} )
            da.to_netcdf(out_dir / f'val_data.encodings.nc')

            # we will save a decoding of the encodings, where one of the latent dimensions is set to zero, for each latent dimension
            for _, peak in enumerate(peak_order):
                enc = oae.encoder(val_data)
                enc[:, peak] = 0
                decoded = oae.decoder(enc)
                decoded = (decoded.detach().numpy() * val_template).unstack('feature').sortby('lat').sortby('lon')
                decoded.to_netcdf(out_dir / f'val_data.decoded.without.{titles2[_]}.nc')

            # now we will save a decoding where none of the latent dims are blanked out
            enc = oae.encoder(val_data)
            decoded = oae.decoder(enc)
            decoded2 = (decoded.detach().numpy() * val_template).unstack('feature').sortby('lat').sortby('lon')
            decoded2.to_netcdf(out_dir / f'val_data.decoded.nc')

            # remove the recreated validation data from the original validation data 
            # we will train the next recursion on this set of 'residuals' 
            val_data = val_data.detach().numpy() - decoded.detach().numpy()

            # now we will remove the recreated training data from the original training data 
            # so the next recursion is trained on the residuals from this AE
            for _ in range(len(training_data)):
                enc_train = oae.encoder(torch.tensor(training_data[_], dtype=torch.float32))
                dec_train = oae.decoder(enc_train)
                training_data[_] = training_data[_] - dec_train.detach().numpy()


            # encode and decode and save test data  for ... something later 
            test_data = torch.tensor(test_data, dtype=torch.float32)
            enc = oae.encoder(test_data)
            decoded = oae.decoder(enc)
            encoded = enc.detach().numpy()

            # flip the signs of the encoded test data according to what we did before! 
            for _, peak in enumerate(peak_order):
                if flips[_]:
                    encoded[:, peak] = encoded[:, peak] * -1 

            timecoord =  [pd.Timestamp(i) for i in test_template.time.values]
            da = xr.DataArray(data=encoded[:, peak_order], name='encodings', dims=('time', 'mode'), coords={'time': timecoord, 'mode':titles2} )
            da.to_netcdf(out_dir / f'test_data.encodings.nc')
            
            decoded2 = (decoded.detach().numpy() * test_template).unstack('feature').sortby('lat').sortby('lon')
            decoded2.to_netcdf(out_dir / f'test_data.decoded.nc')

            # remove decoding of test data from test data for next recursion
            test_data = test_data.detach().numpy() - decoded.detach().numpy()


            # now we handle ERSST
            # we are going to save the residuals in between
            # if we have not recursed yet, we will open the original data and preprocess it 
            all_recurse_outdir = Path(expdir_name) / f'split{split}'
            if recursion == 0: 
                ersst1 = xr.open_dataset(f'~/Desktop/Data/ersstv5/ersstv5.pacific.sst.185401-202501.nc').sst#.sel(lat=slice(-20,20,None)) # * (sst.mean('time') / sst.mean('time')) ##.sortby('lat').sortby('lon')
                ersst, fit, gwm, p  = src.global_detrend(ersst1,deg=2)
                ersst, monthly_clim = src.remove_climo(ersst)
                ersst.name = 'sst'
                ersst.to_netcdf(all_recurse_outdir / "ersstv5.pacific.recursion0.nc")

            ersst1 = xr.open_dataset(all_recurse_outdir / f"ersstv5.pacific.recursion{recursion}.nc").sst
            ersst = ersst1.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature').values 
            ersst = torch.from_numpy(ersst).to(torch.float32)

            ersst_enc = oae.encoder(ersst)
            ersst_dec = oae.decoder(ersst_enc)
            ersst_encoded = ersst_enc.detach().numpy()
            ersst_encoded = ersst_encoded[:, peak_order]
            for _, flip in enumerate(flips): 
                if flip:
                    ersst_encoded[:, _] = -1 * ersst_encoded[:,_]

            da = xr.DataArray(data=ersst_encoded, name='encodings', dims=('time', 'mode'), coords={'time': [pd.Timestamp(i) for i in ersst1.time.values], 'mode': titles2} )
            da.to_netcdf(out_dir / f'ersstv5.encodings{split}.nc')

            ersst_dec = ersst_dec.detach().numpy() * xr.ones_like(ersst1.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature'))
            ts = (ersst1 - ersst_dec.unstack('feature').sortby('lat').sortby('lon'))
            ts.name = 'sst'
            ts.to_netcdf(all_recurse_outdir / f"ersstv5.pacific.recursion{recursion +1}.nc")


            # now we will deal with E3SM
            for mem in range(21):
                if recursion == 0:
                    e3sm1 = xr.open_dataset(f'~/Desktop/Data/e3sm/e3sm.historical.185001-201412.m0{mem:>02}.global.sst.nc').sst.sel(time=slice(sst.time[0], None))#.sel(lat=slice(-20,20,None)) # * (sst.mean('time') / sst.mean('time'))
                    e3sm, fit, gwm, p  = src.global_detrend(e3sm1,deg=2)
                    e3sm, monthly_clim = src.remove_climo(e3sm)
                    e3sm.name = 'sst'
                    e3sm.to_netcdf(all_recurse_outdir / f"e3sm.mem{mem}.pacific.recursion0.nc")

                e3sm1 = xr.open_dataset(all_recurse_outdir / f"e3sm.mem{mem}.pacific.recursion{recursion}.nc").sst
                e3sm = e3sm1.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature').values 
                e3sm = torch.from_numpy(e3sm).to(torch.float32)

                e3sm_enc = oae.encoder(e3sm)
                e3sm_dec = oae.decoder(e3sm_enc)
                e3sm_encoded = e3sm_enc.detach().numpy()
                e3sm_encoded = e3sm_encoded[:, peak_order]
                for _, flip in enumerate(flips): 
                    if flip:
                        e3sm_encoded[:, _] = -1 * e3sm_encoded[:,_]

                da = xr.DataArray(data=e3sm_encoded, name='encodings', dims=('time', 'mode'), coords={'time': [pd.Timestamp(i) for i in e3sm1.time.values], 'mode': titles2} )
                da.to_netcdf(out_dir / 'e3sm' /  f'e3sm.mem{mem}.encodings{split}.nc')

                e3sm_dec = e3sm_dec.detach().numpy() * xr.ones_like(e3sm1.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature'))
                ts = (e3sm1 - e3sm_dec.unstack('feature').sortby('lat').sortby('lon'))
                ts.name='sst'
                ts.to_netcdf(all_recurse_outdir / f"e3sm.mem{mem}.pacific.recursion{recursion +1}.nc")

            # now we will deal with CESM
            for mem in range(21):
                if recursion == 0:
                    e3sm1 = xr.open_dataset(f'~/Desktop/Data/cesm-le/cesm.member0{mem:>02}.global.1p0x1p0.sst.185001-201412.nc').sst#.sel(lat=slice(-20,20,None))
                    e3sm1 = e3sm1.assign_coords({'time': [pd.Timestamp(iii.year, iii.month, iii.day) for iii in e3sm1.coords['time'].values]}).sel(time=slice(sst.time[0], None)) # * (sst.mean('time') / sst.mean('time'))
                    e3sm, fit, gwm, p  = src.global_detrend(e3sm1,deg=2)
                    e3sm, monthly_clim = src.remove_climo(e3sm)
                    e3sm.name = 'sst'
                    e3sm.to_netcdf(all_recurse_outdir / f"cesm.mem{mem}.pacific.recursion0.nc")

                e3sm1 = xr.open_dataset(all_recurse_outdir / f"cesm.mem{mem}.pacific.recursion{recursion}.nc").sst
                e3sm = e3sm1.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature').values 
                e3sm = torch.from_numpy(e3sm).to(torch.float32)

                e3sm_enc = oae.encoder(e3sm)
                e3sm_dec = oae.decoder(e3sm_enc)
                e3sm_encoded = e3sm_enc.detach().numpy()
                e3sm_encoded = e3sm_encoded[:, peak_order]
                for _, flip in enumerate(flips): 
                    if flip:
                        e3sm_encoded[:, _] = -1 * e3sm_encoded[:,_]

                da = xr.DataArray(data=e3sm_encoded, name='encodings', dims=('time', 'mode'), coords={'time': [pd.Timestamp(i) for i in e3sm1.time.values], 'mode': titles2} )
                da.to_netcdf(out_dir / 'cesm' / f'cesm.mem{mem}.encodings{split}.nc')

                e3sm_dec = e3sm_dec.detach().numpy() * xr.ones_like(e3sm1.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature'))
                ts = (e3sm1 - e3sm_dec.unstack('feature').sortby('lat').sortby('lon'))
                ts.name= 'sst'
                ts.to_netcdf(all_recurse_outdir / f"cesm.mem{mem}.pacific.recursion{recursion +1}.nc")


        for recursion in range(M+1):
            Path(all_recurse_outdir / f"ersstv5.pacific.recursion{recursion}.nc").unlink()
            for mem in range(21):
                Path(all_recurse_outdir / f"e3sm.mem{mem}.pacific.recursion{recursion}.nc").unlink()
                Path(all_recurse_outdir / f"cesm.mem{mem}.pacific.recursion{recursion}.nc").unlink()














