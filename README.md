# Knowledge-Guided Machine Learning for Disentangling Pacific Ocean SST Variability across Timescales

This repository contains code for the Knowledge-Guided AutoEncoders (KGAE) described by Hall et al. (2025, submitted/under review). It includes the core `kgae` package, data-preparation scripts, and analysis scripts used for the manuscript.

The data pipeline in `data_pipeline/` contains scripts used to access and preprocess ERA5 and ORAS5 data. CESM2 and E3SMv2 model data were retrieved from NCAR Derecho `/glade/` storage and are publicly available as indicated in Hall et al. (2025).

The primary KGAE implementation is in `kgae/`. Analysis and figure-generation scripts are in `scripts/`, `final_scripts/`, and `publication_scripts/`.

Precomputed KGAE encodings for basin-wide and tropical experiments are available in `KGAE_encodings.nc`.

This repository is a curated research-code snapshot for reproducing the analyses associated with the manuscript. Some scripts expect external data files or local paths to be configured before running.
