# Quantized Time Series Modeling for Financial Data

This project was created for research purposes at the Financial Computing Lab of the Budapest University of Technology and Economics.

## Overview

This project implements a complete machine learning pipeline for financial time series analysis and prediction. The focus is on **quantization-based preprocessing**, **categorical label generation**, and **deep learning models** (LSTM networks) for sequential data prediction.

It includes:
- Data loading and processing utilities.
- Quantization techniques for transforming continuous financial data into categorical features.
- LSTM model architecture for training on historical financial time series.
- Backtesting scripts for model evaluation.

## Structure

- **`data_processing.py`** – Utilities for reading CSV data and adding quantized labels.
- **`quantization.py`** – Quantization logic for converting continuous targets into discrete categories.
- **`x_y_arrays.py`** – Generates training and test sets, builds sequences for model input.
- **`model.py`** – Defines the LSTM model architecture.
- **`train.py`** – Training script for fitting the model.
- **`backtest.py`** – Model backtesting and evaluation script.

## Research Context

This project was developed as part of one of my introduction research at the Budapest University of Technology and Economics (BME) in the field of Financial Computing. The goal is to explore novel approaches for quantizing financial time series and leveraging deep learning models for improved prediction and strategy development.

## Codebase Description

This project is a modular Python pipeline designed for **quantized time series modeling** of financial data. It combines **quantization techniques** with an **LSTM neural network** to predict categorical movements in asset prices. Here’s how it works, step-by-step:

### 1️⃣ Data Processing (`data_processing.py`)
- Loads historical financial data (e.g., asset prices from a CSV).
- Normalizes and preprocesses the data.
- Generates **quantized labels** using a specified number of bits (`n_bits`). For example, continuous price changes can be categorized into 40 discrete bins.

### 2️⃣ Quantization (`quantization.py`)
- Provides the logic for converting continuous targets (like asset returns or prices) into **discrete categories**.
- These categories serve as **classification labels** for the model, enabling the prediction of movements in a quantized form.

### 3️⃣ Sequence Generation (`x_y_arrays.py`)
- Transforms the preprocessed data into **sequences** suitable for time series modeling.
- Implements a sliding window approach, creating **X (features)** and **Y (labels)** for supervised learning.
- Splits the data into **training** and **test** sets.

### 4️⃣ Model Definition (`model.py`)
- Defines an **LSTM-based neural network** using TensorFlow/Keras.
- The model is designed to predict **probabilities of each quantized class** over the input sequences.

### 5️⃣ Training (`train.py`)
- Trains the LSTM model on the prepared data.
- Configurable parameters like `n_bits`, `window size`, `batch size`, and `epochs` allow experimentation.

### 6️⃣ Backtesting (`backtest.py`)
- Uses the trained model to predict quantized class probabilities on test data.
- Outputs predictions and can be extended to **evaluate model performance** and **simulate trading strategies**.
- For the 4H timeframe I found these parameters to be succesful:
    - N_DATAPOINTS = 2000
    - N_TESTPOINTS = 500
    - WINDOW = 3
    - TP/SL = 0.002/0.002 (In real time, ATR might be better)
    - N_BITS = 25
---

### What does it do?

- **Transforms raw financial data** into a **quantized categorical format** suitable for classification.
- Trains a deep learning model (LSTM) to **learn temporal dependencies** and **predict future categories** in financial time series.
- Provides a foundation for developing **probabilistic trading strategies**, **forecasting models**, or further research in **discretized financial modeling**.
