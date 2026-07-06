from .utilities import deniell, crosscorrelation_by_month, crosscorrelation_all, detrend, calc_wnl,  open_oni, open_pdo, open_npi, remove_climo, remove_directory, global_detrend,  multivariate_linear_regression_with_significance
from .linalg_solve import mps_linalg_solve 
from .kgae import KGAE
from .wave_filter import wave_filter, low_pass, high_pass, band_pass
from .power_spectra import compute_smoothed_power_spectra
from .tendencies import  jacobian,  regress_tendencies, bootstrap_tendency_regression, bootstrap_ci, smooth_spatial, regress_tendencies_r2
from .visualize import plot_alpha_beta_maps, plot_grid_maps 
from .composite import conditional_composite_alpha_beta, cross_composites, cross_composite_diffs
from .plot_alphabeta import plot_geometry_cv
from .plot_histograms import plot_mode_histograms, plot_joint_histograms
from .results import open_alphabetas, open_latents, open_reconstructions, open_references
from .ema import EMA 
from .compute_patterns import compute_foldmean_decoder_probes
__version__ = "KGAE 0.0.1"