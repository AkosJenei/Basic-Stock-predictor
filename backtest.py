# backtest.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from train import model
from data_processing import DataProcessor

# -----------------------------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------------------------
CSV_PATH        = "historical_data/XAUUSD_1h_historical_data.csv"
N_DATAPOINTS    = 50000    # total rows in this slice
N_TESTPOINTS    = 500     # last 500 rows for backtest
WINDOW          = 3
OFFSET          = 49000       # start row of the 5k-slice in the full CSV

INITIAL_CAP     = 5_000.0
LEVERAGE        = 100.0
RISK_PER_TRADE  = 0.30
STOP_LOSS_PCT   = 1#0.002
TAKE_PROFIT_PCT = 1#0.002

QUANTIZER_PATH  = "quantizer.pkl"

# -----------------------------------------------------------------------------
# 1) Load the 5k-row slice [OFFSET : OFFSET+5000]
# -----------------------------------------------------------------------------
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes = df["Close"].values
highs   = df["High"].values
lows    = df["Low"].values

# -----------------------------------------------------------------------------
# 2) Load pre-fitted quantizer & transform closes → labels → one-hot
# -----------------------------------------------------------------------------
with open(QUANTIZER_PATH, "rb") as f:
    quantizer = pickle.load(f)

labels  = quantizer.transform(closes)
one_hot = np.eye(quantizer.get_bits(), dtype=int)[labels]

# split sizes
TRAIN_SIZE = N_DATAPOINTS - N_TESTPOINTS
start_idx  = TRAIN_SIZE + WINDOW   # first index where we backtest

# -----------------------------------------------------------------------------
# 3) Real-time backtest loop on last 500 rows
# -----------------------------------------------------------------------------
capital       = INITIAL_CAP
equity_curve  = [capital]
trade_returns = []
signals_list  = []
prev_label    = None

for t in range(start_idx, N_DATAPOINTS - 1):
    X_in   = one_hot[t-WINDOW:t][np.newaxis, :, :]
    y_prob = model.predict(X_in, verbose=0)[0]
    lbl    = int(np.argmax(y_prob))

    # generate +1/−1/0 signal
    if prev_label is None:
        signal = 0
    else:
        signal = +1 if lbl > prev_label else -1 if lbl < prev_label else 0

    signals_list.append(signal)
    prev_label = lbl

    # execute trade when signal ≠ 0
    if signal != 0:
        entry = closes[t]
        tp    = entry*(1+TAKE_PROFIT_PCT) if signal>0 else entry*(1-TAKE_PROFIT_PCT)
        sl    = entry*(1-STOP_LOSS_PCT)   if signal>0 else entry*(1+STOP_LOSS_PCT)

        hi, lo, nxt = highs[t+1], lows[t+1], closes[t+1]
        if   signal>0 and hi  >= tp: exit_price = tp
        elif signal<0 and lo  <= tp: exit_price = tp
        elif signal>0 and lo  <= sl: exit_price = sl
        elif signal<0 and hi  >= sl: exit_price = sl
        else:                          exit_price = nxt

        margin   = capital * RISK_PER_TRADE
        notional = margin * LEVERAGE
        units    = notional / entry
        pnl      = units * ((exit_price-entry) if signal>0 else (entry-exit_price))

        capital += pnl
        trade_returns.append(pnl / margin)

    equity_curve.append(capital)

equity_curve  = np.array(equity_curve)
trade_returns = np.array(trade_returns)

# -----------------------------------------------------------------------------
# 4) Metrics & Reporting
# -----------------------------------------------------------------------------
test_labels     = labels[TRAIN_SIZE:]
true_dirs_full  = np.sign(np.diff(test_labels))
true_dirs       = true_dirs_full[WINDOW:]      # drop first WINDOW so lengths match
pred_dirs       = np.array(signals_list)

cm = confusion_matrix(true_dirs, pred_dirs, labels=[-1,0,1])
print("=== Directional Accuracy ===")
print("Confusion Matrix (-1,0,+1):")
print(cm)
print("\nClassification Report:")
print(classification_report(true_dirs, pred_dirs, labels=[-1,0,1]))

cum_return = equity_curve[-1]/INITIAL_CAP - 1
max_dd     = (equity_curve/np.maximum.accumulate(equity_curve) - 1).min()
print(f"\n=== Performance Summary ===")
print(f"Cumulative Return: {cum_return:.2%}")
print(f"Max Drawdown:     {max_dd:.2%}")

# -----------------------------------------------------------------------------
# 5) Plots
# -----------------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(equity_curve, label="Equity Curve")
plt.title("Equity Curve"); plt.grid(); plt.legend()

plt.figure(figsize=(10,4))
plt.plot(equity_curve/np.maximum.accumulate(equity_curve) - 1,
         color="red", label="Drawdown")
plt.title("Drawdown Curve"); plt.grid(); plt.legend()

plt.figure(figsize=(8,4))
plt.hist(trade_returns, bins=50)
plt.title("Distribution of Trade Returns"); plt.grid()

plt.show()
