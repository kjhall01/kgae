# Center-Action Bimodality Artifact Discovery

## Search Roots
- `.`
- `final_scripts`
- `publication_scripts`
- `scripts`
- `data_pipeline`
- `/Users/kylehall/Desktop/codex-dev/kgae_fig`

## Selected Inputs
- sst: `data_pipeline/era5.sst.pacific.1x1.1940-2023.nc`
- latents: `final_scripts/era5.latents.v1.nc`
- patterns: `scripts/cache_alpha_and_linear_composites.nc`
- config: `analysis/center_action_boxes.yaml`

## Processing Notes
- Optional sklearn GaussianMixture unavailable; GMM BIC metrics omitted (numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject).
- Loaded manual center-of-action boxes from `analysis/center_action_boxes.yaml`.
- Selected `encodings` as the latents variable.
- Averaged latent dimension `seed` to form an ensemble mean.
- Selected `sst` as the sst variable.
- Prepared SSTA with the same steps as kgae.global_detrend(..., deg=2) followed by kgae.remove_climo(...): subtract the cosine-latitude-weighted Pacific mean quadratic trend, then subtract the monthly climatology over the loaded/common period.
- Selected `alpha_mean` as the patterns variable.
- Loaded top-row mode patterns from `/Users/kylehall/Desktop/codex-dev/kgae/scripts/cache_alpha_and_linear_composites.nc`.
- Decadal: aligned by exact datetime coordinates over 900 samples (1940-01-01 to 2014-12-01).
- Interannual: aligned by exact datetime coordinates over 900 samples (1940-01-01 to 2014-12-01).
- Quasibiennial: aligned by exact datetime coordinates over 900 samples (1940-01-01 to 2014-12-01).

## SST Candidates
- `data_pipeline/era5.sst.pacific.1x1.1940-2023.nc`

## Latent Candidates
- `KGAE_encodings.nc`
- `final_scripts/cesm.latents.v1.nc`
- `final_scripts/e3sm.latents.v1.nc`
- `final_scripts/era5.latents.v1.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed0/test_latents.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed0/xval_latent.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed1/test_latents.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed1/xval_latent.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed2/test_latents.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed2/xval_latent.nc`
- `publication_scripts/cesm.latents.v1.nc`
- `publication_scripts/e3sm.latents.v1.nc`
- `publication_scripts/era5.latents.v1.nc`
- `scripts/debug_latents.nc`

## Pattern Candidates
- `final_scripts/alpha_full_pct-lt-zero_bootstrap=1000_ensemble=100.nc`
- `final_scripts/alpha_full_pct-lt-zero_bootstrap=100_ensemble=1.nc`
- `final_scripts/alpha_full_pct-lt-zero_bootstrap=100_ensemble=3.nc`
- `final_scripts/alpha_seed.nc`
- `final_scripts/beta_full_pct-lt-zero_bootstrap=1000_ensemble=100.nc`
- `final_scripts/beta_full_pct-lt-zero_bootstrap=100_ensemble=1.nc`
- `final_scripts/beta_full_pct-lt-zero_bootstrap=100_ensemble=3.nc`
- `final_scripts/beta_seed.nc`
- `final_scripts/climate_mode_patterns.nc`
- `final_scripts/corr_outputs/corr_alpha_mask-none.nc`
- `final_scripts/corr_outputs/corr_beta_mask-none.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed0/full_alpha.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed0/full_beta.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed0/xval_alpha.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed0/xval_beta.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed1/full_alpha.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed1/full_beta.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed1/xval_alpha.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed1/xval_beta.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed2/full_alpha.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed2/full_beta.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed2/xval_alpha.nc`
- `final_scripts/large-ensemble-3-1940-2014/seed2/xval_beta.nc`
- `publication_scripts/supplemental/climate_mode_patterns.nc`
- `scripts/cache_alpha_and_linear_composites.nc`
- `scripts/cache_betas_composites_regression.nc`
- `scripts/climate_mode_spatial_patterns.nc`
- `scripts/ensemble_bootstrap_composites_alpha.nc`
- `scripts/ensemble_bootstrap_composites_alpha_sig.nc`
- `scripts/ensemble_bootstrap_composites_beta.nc`
- ... 1 more omitted

## Relevant Figure Candidates
- `final_scripts/alphas_betas_final.png`
- `final_scripts/climate_modes.png`
- `final_scripts/composite_grid_alpha_Ji_selfmag_interactionterm.png`
- `final_scripts/decoder_patterns.png`
- `final_scripts/decoder_probe_crossmode.png`
- `final_scripts/decoder_surrogate_train_r2.png`
- `final_scripts/fold-sensitivity-decoder-probes_refIsFoldMean.png`
- `final_scripts/progressive_split_figure_final.pdf`
- `final_scripts/progressive_split_figure_final.png`
- `publication_scripts/figures/figure1_schematics.pdf`
- `publication_scripts/figures/figure2_timeseries.pdf`
- `publication_scripts/figures/figure3_lbvfds.pdf`
- `publication_scripts/figures/figure4_composites.pdf`
- `publication_scripts/figures/figure5_crossmode.pdf`
- `publication_scripts/figures/figure6_qb-ia.pdf`
- `publication_scripts/figures/figure7_ia-de.pdf`
- `publication_scripts/figures/figure8_progressive_split.pdf`
- `publication_scripts/figures/figure9_model-eval.pdf`
- `publication_scripts/supplemental/climate_modes.pdf`
- `publication_scripts/supplemental/fold-sensitivity-decoder-probes_refIsFoldMean.pdf`
- `publication_scripts/supplemental/large-ens-approximation-direct-decoder-probes.pdf`
- `publication_scripts/supplemental/large-ens-approximation-direct-decoder-probes.png`
- `publication_scripts/supplemental/split-half-experiment-direct-decoder-probes.pdf`
- `scripts/alphas_and_composites_final.png`
