# Knowledge-Guided Machine Learning for Disentangling Pacific Ocean SST Variability across Timescales

Dear reader, 

This repository contains code implementing the Knowledge-Guided AutoEncoders (KGAE) described by Hall et. al., 2025 (submitted/under review), as well as the data pipeline (in /data_pipeline) used to access ERA5 and ORAS5 data used to train them. Model Data from CESM2 and E3SMv2 were retrieved from /glade/ storage through NCAR's Derecho compute system, and are available publicly as indicated in Hall et al 2025. 

The primary implementation of KGAE is in src/network.py. The scripts used to 1) run all the recursive, cross-validated experiments and 2) produce the figures in Hall et. al., 2025 are available in /scripts 

KGAE primary, secondary, and tertiary modes of variability from both basin-wide and tropical experiments are available in KGAE_encodings.nc (the test period for the tropical experiment suffers from some tech debt, all the analysis was on the crossvalidated training period anyway) 

This is a small (albeit critically important, minimum viable product) sampling of the scripts/code/analysis we did for this project- in the interest of clarity we removed anything unrelated to the final paper. An unfortunate side effect is that nothing here will work out-of-the-box - perhaps one day I'll assemble a Jupyter notebook tailored for this repo. 

Best wishes,
Kyle

