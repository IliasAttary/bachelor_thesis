# Autoencoder Embeddings for Adaptive Model Selection and Drift Detection (AE-AMS)

This repository contains the code for my bachelor’s thesis **AE-AMS**, a framework for adaptive model selection in time series forecasting. The approach combines convolutional autoencoder representations with region-of-competence–based model selection to dynamically choose forecasting models.

## Core idea

AE-AMS uses autoencoder embeddings as the feature space in which similarities between time-series windows are measured. Windows from a validation set are encoded using a convolutional autoencoder and assigned to the forecasting model that performs best on them, forming regions of competence. During online forecasting, the current window is encoded and compared to these regions in latent space to select the most suitable model.

The forecasting pool includes classical statistical methods, linear and kernel-based regressors, tree-based ensembles, and multiple neural architectures such as MLPs, LSTMs, BiLSTMs, and CNN–LSTM hybrids.

## Repository layout

```text
.
├── data/                     # Dataset loaders (univariate)
├── forecasters/              # Forecasting models (statistical & ML)
├── autoencoder.py            # 1D convolutional autoencoder (PyTorch)
├── drift_detectors.py        # Hoeffding-style drift detectors
├── utils.py                  # Windowing, normalization, distances, helpers
└── *.ipynb                   # Experiment notebooks
