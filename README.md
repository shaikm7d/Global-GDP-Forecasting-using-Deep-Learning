# 🌐 Global GDPpc Forecasting using Deep Learning
**Author:** Shaik Mohammed

## 📌 Project Overview

World Bank GDP per-capita tells a long story about how countries grow but it’s messy, uneven, and hard to predict. This project follows that story from raw data to future forecasts and analyses.

In this project:
- Pulled data using the **wbdata API**, then **preprocessed** it by handling missing values and performing **EDA** to understand trends, distributions, and patterns.
- Expanded the dataset by computing summary statistics (**mean, median, max, variance**) for each indicator, then trained an **Autoencoder (AE)** and applied **K-Means** on its **latent space** to discover hidden economic similarity groups.
- Binned **GDP per-capita (GDPpc)** into four development tiers: **['under-developed', 'developing', 'emerging', 'developed']**, and trained an **MLP** to predict the development label.
- Built **time-series windows** and trained **LSTM**, **CNN-LSTM**, and **Transformer** models to perform **multi-step GDPpc forecasting**.
- Trained a **Variational Autoencoder (VAE)** to learn the overall data distribution and generate synthetic samples, then **augmented the training data** and retrained the best model (**LSTM**) to improve **generalization**.
