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

## Historical data

Historical data files, can be found in the historical_data folder. This strategy can be tested on various forex currencies, and commodities.The 4 Hour timeframe seem to work well, the parameters for that will be shown in the backtesting section.

## Research Context

This project was developed as part of one of my introductory research at the Budapest University of Technology and Economics (BME) in the field of Financial Computing. The goal is to explore novel approaches for quantizing financial time series and leveraging deep learning models for improved prediction and strategy development.

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

---

### Results:

## With Take Profit and Stop Loss
## XAUUSD(Gold/USD) First case
** Confusion Matrix:
![XAUUSD-conf-matrix-1](https://github.com/user-attachments/assets/9b79705c-a9f6-4880-a24b-2a1adba70a1f)

** Equity curve:
![XAUUSD-equity-1](https://github.com/user-attachments/assets/aceb0aff-96ce-48ad-abcb-83450329e8fc)

** Drawdown curve:
![XAUUSD-drawdown-1](https://github.com/user-attachments/assets/bbe861ea-9889-458e-a141-abaa285f4408)

** Distribution of trades:
![XAUUSD-dist-1](https://github.com/user-attachments/assets/16950681-9769-48f9-b031-215e66eea49a)

## XAUUSD(Gold/USD) Second case
**Confusion Matrix:
![XAUUSD-conf-matrix-3](https://github.com/user-attachments/assets/023f6929-9302-4a6f-8745-bc080bd8bbff)

** Equity curve:
![XAUUSD-equity-3](https://github.com/user-attachments/assets/ec34d8a6-7c11-4a0e-ab9b-101730cf37de)

** Drawdown curve:
![XAUUSD-drawdown-3](https://github.com/user-attachments/assets/5a60d5b9-421b-436a-a881-fbfaf1b93351)

** Distribution of trades:
![XAUUSD-dist-3](https://github.com/user-attachments/assets/c8b6125f-d397-4145-b94b-16f4d1d4d0ad)

## AUDCAD
** Confusion Matrix:
![AUDCAD-conf-matrix-1](https://github.com/user-attachments/assets/8fc7eea1-c77c-4242-95b5-ee1db40f37ca)

** Equity curve:
![AUDCAD-equity-1](https://github.com/user-attachments/assets/1e57961c-3f0d-4c23-b1a4-723bcbf65fc0)

** Drawdown curve:
![AUDCAD-drawdown-1](https://github.com/user-attachments/assets/2e140eaf-5254-44cf-955c-2da945fd9c9d)

** Distribution of trades:
![AUDCAD-dist-1](https://github.com/user-attachments/assets/8abe1efb-eb45-4905-9045-d0ebf489bc9d)

## USDCHF
**Confusion Matrix:
![USDCHF-conf-matrix-1](https://github.com/user-attachments/assets/9e4c0b6d-cd92-49bc-a8be-3c8f46634dd7)

** Equity curve:
![USDCHF-equity-1](https://github.com/user-attachments/assets/02161844-6625-4c08-ba7d-02df2349c87f)

** Drawdown curve:
![USDCHF-drawdown-1](https://github.com/user-attachments/assets/703fd589-40f1-4d96-8546-9ba09f3c4f3b)

** Distribution of trades:
![USDCHF-dist-1](https://github.com/user-attachments/assets/0b1a5dc5-7040-4c8e-a2e0-0a3ff975e07f)
