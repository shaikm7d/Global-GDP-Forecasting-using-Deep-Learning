# 🌐 Global GDPpc Forecasting using Deep Learning
**Author:** Shaik Mohammed

## 📌 Project Overview

World Bank GDP per-capita tells a long story about how countries grow but it’s messy, uneven, and hard to predict. This project follows that story from raw data to future forecasts and analyses.

In this project:
- Pull data using the **wbdata API**, then **preprocess** it by handling missing values and performing **EDA** to understand trends, distributions, and patterns.
- Expanded the dataset by computing summary statistics (**mean, median, max, variance**) for each indicator, then trained an **Autoencoder (AE)** and applied **K-Means** on its **latent space** to discover hidden economic similarity groups.
- Binned **GDP per-capita (GDPpc)** into four development tiers: **['under-developed', 'developing', 'emerging', 'developed']**, and trained an **MLP** to predict the development label.
- Built **time-series windows** and trained **LSTM**, **CNN-LSTM**, and **Transformer** models to perform **multi-step GDPpc forecasting**.
- Trained a **Variational Autoencoder (VAE)** to learn the overall data distribution and generate synthetic samples, then **augmented the training data** and retrained the best model (**LSTM**) to improve **generalization**.

## 📂 Dataset & Code Files

> Recommended order: **1 → 2 → 3 → 4 → 5**

| Step | File | Purpose | Key Output |
|------|------|---------|------------|
| 1 | `Data_Download_Preprocessing_EDA.ipynb` + `data_download_preprocessing.py` | Pull World Bank indicators via **wbdata API**, clean/preprocess (missing values), and run **EDA** (incl. correlation matrix). | `world_bank_data_clean.csv` |
| 2 | `AE_Clustering.ipynb` | Train **Autoencoder** and run **K-Means** on the **latent space** to identify economic similarity clusters. | Cluster labels / latent embeddings |
| 3 | `MLP.ipynb` | Bin GDPpc into 4 development tiers and train an **MLP** to predict the development label. | Classification metrics + confusion matrix |
| 4 | `Time_Series_Forecasting.ipynb` + (`LSTM.py`, `CNN_LSTM.py`, `Transformer.py`) | Build time-series windows and benchmark **LSTM / CNN-LSTM / Transformer** for multi-step GDPpc forecasting. | Forecast metrics + plots |
| 5 | `VAE_Augment_LSTM.ipynb` + `LSTM.py` | Train a **VAE** to generate synthetic samples, augment training data, and retrain the best model (**LSTM**) to improve generalization. | Augmented training results + plots |

### 📄 Main Dataset
- **`world_bank_data_clean.csv`** - cleaned World Bank panel dataset generated in **Step 1** and used throughout the project.
