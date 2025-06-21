import pickle
import numpy as np
import matplotlib.pyplot as plt

from train import model
from data_processing import DataProcessor
from quantization import build_codebook

CSV_PATH = "historical_data/AUDCHF_15m_historical_data.csv"
N_DATAPOINTS = 500
N_TESTPOINTS = 300
WINDOW = 3
OFFSET = 71500
INITIAL_CAP = 10_000.0
QUANTIZER_PATH = "quantizer.pkl"
USE_PRICE_CHANGES = False

# Load data
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes = df["Close"].values
price_changes = dp.get_price_changes()
series = price_changes if USE_PRICE_CHANGES else closes

# Load quantizer and labels
with open(QUANTIZER_PATH, "rb") as f:
    quantizer = pickle.load(f)
labels = quantizer.transform(series)
M = quantizer.get_bits()
S = build_codebook(M)

one_hot = np.eye(M, dtype=np.float32)[labels]
start_idx = len(labels) - N_TESTPOINTS

capital = INITIAL_CAP
equity_curve = [capital]

for t in range(start_idx, len(labels) - 1):
    x_in = one_hot[t-WINDOW:t].reshape(1, -1)
    out = model.predict(x_in, verbose=0)[0]
    bits = np.where(out >= 0, 1, -1)
    curr_label = labels[t]
    in_lower = S[0, curr_label] == 1
    if np.all(bits == -1) and in_lower:
        pos = 1
    elif np.all(bits == 1) and not in_lower:
        pos = -1
    else:
        pos = 0
    pnl = (closes[t+1] - closes[t]) * pos
    capital += pnl
    equity_curve.append(capital)

plt.figure(figsize=(10, 4))
plt.plot(equity_curve)
plt.title("Equity Curve")
plt.grid()
plt.show()

