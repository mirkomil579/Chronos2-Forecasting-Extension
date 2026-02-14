# 📈 Deep NLP Project: Chronos-2 Time Series Forecasting

**Course:** Deep Natural Language Processing (Master's Degree)  
**Assigned Paper:** *Chronos-2: From Univariate to Universal Forecasting* **Report Link:** [Insert your Overleaf/PDF Link Here]  

## 🎯 Project Overview
This repository contains the code and experiments for reproducing and extending the findings of the Chronos-2 paper. Chronos-2 frames time series forecasting as a language modeling problem, utilizing a T5-based architecture to predict quantized continuous values as discrete tokens. 

The primary goal of this project is to evaluate the model's touted "Universal Zero-Shot" capabilities and compare them directly against a domain-adapted (fine-tuned) version to assess whether foundation models still require targeted weight updates for optimal performance.

## 🚀 The Extension (Domain Adaptation)
To satisfy the project extension requirements, this repository goes beyond the base paper by conducting a **Fine-Tuning vs. Zero-Shot Domain Adaptation Analysis**. 
* **Zero-Shot Baseline:** Running the pre-trained `amazon/chronos-t5-tiny` model with frozen weights.
* **Domain Adaptation:** Unfreezing the weights and fine-tuning the model on a specific time-series dataset.
* **Evaluation:** Comparing the Mean Absolute Scaled Error (MASE) to quantify the performance gain achieved through domain adaptation.

## 📂 Repository Structure
```text
├── data/                    # Directory for datasets (downloaded via script)
├── notebooks/               
│   └── chronos_experiments.ipynb  # Core reproducible experiment notebook
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
