# Center-of-Action Bimodality Check Response Note

We added a simple center-of-action diagnostic to address the reviewer concern that bimodality in learned latent-coordinate PDFs may be induced partly by the KGAE representation geometry rather than by directly bimodal SST.

For each learned mode, the diagnostic computes an area-weighted SST anomaly index from fixed positive and negative center-of-action boxes aligned with the paper maps. When a mode has multiple same-sign boxes, the script first forms the union of those grid cells. The index is positive-box-union mean SSTA minus negative-box-union mean SSTA. This is intentionally a box-average sanity check, not a full spatial projection onto the KGAE pattern.

The figure is arranged with the KGAE mode/linear-response pattern in the top row and the corresponding latent-coordinate versus SST-box-index PDFs in the bottom row. The boxes shown on the top-row maps are the same boxes used to compute the bottom-row SST indices.

The SST anomaly preprocessing follows the same sequence used in the other analysis scripts: `kgae.global_detrend(..., deg=2)` style removal of the cosine-latitude-weighted Pacific mean quadratic trend, followed by `kgae.remove_climo(...)` style monthly climatology removal over the loaded common period.

## Inputs
- sst: `data_pipeline/era5.sst.pacific.1x1.1940-2023.nc`
- latents: `final_scripts/era5.latents.v1.nc`
- patterns: `scripts/cache_alpha_and_linear_composites.nc`
- config: `analysis/center_action_boxes.yaml`

## Box Definitions and Distribution Metrics

| mode          | positive_box                                                   | negative_box                    |   n_valid_samples |   latent_skewness |   latent_kurtosis_pearson |   latent_bimodality_coefficient | latent_gmm_bic_1_minus_2   |   box_skewness |   box_kurtosis_pearson |   box_bimodality_coefficient | box_gmm_bic_1_minus_2   | box_source                                       |
|:--------------|:---------------------------------------------------------------|:--------------------------------|------------------:|------------------:|--------------------------:|--------------------------------:|:---------------------------|---------------:|-----------------------:|-----------------------------:|:------------------------|:-------------------------------------------------|
| Decadal       | lat[-5.5,5.5], lon[-35.5,-5.5]; lat[45.5,60.5], lon[0.5,45.5]  | lat[25.5,45.5], lon[-35.5,25.5] |               900 |            -0.1   |                     1.645 |                           0.614 |                            |          0.009 |                  2.541 |                        0.394 |                         | manual config: analysis/center_action_boxes.yaml |
| Interannual   | lat[-5.5,5.5], lon[0.5,100.5]                                  | lat[24.5,44.5], lon[11.5,51.5]  |               900 |             0.211 |                     2.19  |                           0.477 |                            |          0.843 |                  3.983 |                        0.429 |                         | manual config: analysis/center_action_boxes.yaml |
| Quasibiennial | lat[-5.5,5.5], lon[30.5,100.5]; lat[45.5,60.5], lon[15.5,60.5] | lat[28.5,48.5], lon[-25.5,15.5] |               900 |             0.207 |                     2.966 |                           0.352 |                            |          0.581 |                  3.608 |                        0.371 |                         | manual config: analysis/center_action_boxes.yaml |

## Interpretation

The box-index and latent-coordinate distribution metrics are mixed. The diagnostic should be interpreted as a sanity check rather than evidence for or against discrete physical regimes.

In all cases, the latent-coordinate PDFs should be described as PDFs of learned coordinates. They are not direct PDFs of physical SST variables, even when the corresponding box-average SST indices show similar distributional structure.

## Reproducibility Notes
- Optional sklearn GaussianMixture unavailable; GMM BIC metrics omitted (numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject).
- Loaded manual center-of-action boxes from `analysis/center_action_boxes.yaml`.
- Selected `encodings` as the latents variable.
- Averaged latent dimension `seed` to form an ensemble mean.
- Selected `sst` as the sst variable.
- Prepared SSTA with the same steps as kgae.global_detrend(..., deg=2) followed by kgae.remove_climo(...): subtract the cosine-latitude-weighted Pacific mean quadratic trend, then subtract the monthly climatology over the loaded/common period.
- Selected `alpha_mean` as the patterns variable.
- Loaded top-row mode patterns from `~/Desktop/codex-dev/kgae/scripts/cache_alpha_and_linear_composites.nc`.
- Decadal: aligned by exact datetime coordinates over 900 samples (1940-01-01 to 2014-12-01).
- Interannual: aligned by exact datetime coordinates over 900 samples (1940-01-01 to 2014-12-01).
- Quasibiennial: aligned by exact datetime coordinates over 900 samples (1940-01-01 to 2014-12-01).
