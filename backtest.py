import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from data_processing import DataProcessor
from train import *

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CSV_PATH       = "historical_data/XAUUSD_15m_historical_data.csv"
N_DATAPOINTS   = 10000
N_BITS         = 200
WINDOW         = 3
TRAIN_RATIO    = 0.7

INITIAL_CAP    = 5_000.0
LEVERAGE       = 100.0
RISK_PER_TRADE = 0.30
HOLD_BARS      = 1

BATCH_SIZE     = 64
EPOCHS         = 100
LR             = 2.85e-4
PATIENCE       = 10

# -----------------------------------------------------------------------------
# 1) Load & preprocess
# -----------------------------------------------------------------------------
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, N_DATAPOINTS)
closes = df["Close"].values

# quantize & one‐hot
labels = dp.add_quantized_labels(n_bits=N_BITS)
one_hot = np.eye(N_BITS, dtype=int)[labels]  # shape (T, N_BITS)

# -----------------------------------------------------------------------------
# 3) Predict & generate signals
# -----------------------------------------------------------------------------
y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# simple directional signal: compare to previous label
signals = np.zeros_like(y_pred_labels, dtype=int)
for t in range(1, len(y_pred_labels)):
    if y_pred_labels[t] > y_pred_labels[t - 1]:
        signals[t] = +1
    elif y_pred_labels[t] < y_pred_labels[t - 1]:
        signals[t] = -1
    # else 0

# align prices for test period
# X_test start corresponds to bar index = WINDOW + split
idx_offset = WINDOW + len(X_train)
entry_idxs = np.arange(idx_offset, idx_offset + len(signals) - HOLD_BARS)
exit_idxs  = entry_idxs + HOLD_BARS

# -----------------------------------------------------------------------------
# 4) Simulate execution / portfolio
# -----------------------------------------------------------------------------
capital = INITIAL_CAP
equity_curve = [capital]
trade_returns = []

for i, sig in enumerate(signals[:-HOLD_BARS]):
    if sig == 0:
        equity_curve.append(capital)
        continue

    entry_price = closes[entry_idxs[i]]
    exit_price  = closes[exit_idxs[i]]

    # margin used and position size
    margin     = capital * RISK_PER_TRADE
    notional   = margin * LEVERAGE
    units      = notional / entry_price

    # pnl
    if sig == +1:
        pnl = units * (exit_price - entry_price)
    else:
        pnl = units * (entry_price - exit_price)

    capital += pnl
    equity_curve.append(capital)
    trade_returns.append(pnl / (margin))  # return on margin

equity_curve = np.array(equity_curve)
trade_returns = np.array(trade_returns)

# -----------------------------------------------------------------------------
# 5) Metrics & Analytics
# -----------------------------------------------------------------------------
# timing
n_bars = len(equity_curve)
total_yrs = (n_bars * 15) / (60 * 24 * 365)

cum_return = equity_curve[-1] / INITIAL_CAP - 1
ann_return = (1 + cum_return) ** (1 / total_yrs) - 1

# volatility & Sharpe/Sortino
periods_per_year = len(trade_returns) / total_yrs
vol = np.std(trade_returns) * np.sqrt(periods_per_year)
down_std = np.std(trade_returns[trade_returns < 0]) * np.sqrt(periods_per_year)
sharpe = ann_return / vol if vol != 0 else np.nan
sortino = ann_return / down_std if down_std != 0 else np.nan

# drawdown
peak = np.maximum.accumulate(equity_curve)
drawdown = equity_curve / peak - 1
max_dd = drawdown.min()
dd_dur = np.max(np.diff(np.where(drawdown == 0)[0])) if any(drawdown == 0) else len(drawdown)

# trade‐level stats
wins   = trade_returns[trade_returns > 0]
losses = trade_returns[trade_returns < 0]
win_rate      = len(wins) / len(trade_returns) if len(trade_returns) else np.nan
avg_win       = wins.mean()   if len(wins)   else 0
avg_loss      = losses.mean() if len(losses) else 0
profit_factor = wins.sum() / (-losses.sum()) if losses.sum() != 0 else np.inf
expectancy    = win_rate * avg_win + (1 - win_rate) * avg_loss

# classification report on directional accuracy
# true direction: compare next bar label vs current test label
true_dirs = []
for t in range(1, len(Y_test)):
    true_dirs.append(
        1 if Y_test[t] > Y_test[t - 1]
        else -1 if Y_test[t] < Y_test[t - 1]
        else 0
    )
true_dirs = np.array(true_dirs)
pred_dirs = signals[1:]
acc_dir   = (pred_dirs == true_dirs).mean() if len(true_dirs) else np.nan
cm        = confusion_matrix(true_dirs, pred_dirs, labels=[-1,0,1])
clf_report = classification_report(true_dirs, pred_dirs, labels=[-1,0,1])

# -----------------------------------------------------------------------------
# 6) Reporting
# -----------------------------------------------------------------------------
print("\n=== Performance Summary ===")
print(f"Cumulative Return:      {cum_return:.2%}")
print(f"Annualized Return:      {ann_return:.2%}")
print(f"Annualized Volatility:  {vol:.2%}")
print(f"Sharpe Ratio:           {sharpe:.2f}")
print(f"Sortino Ratio:          {sortino:.2f}")
print(f"Max Drawdown:           {max_dd:.2%} over {int(dd_dur)} bars")
print("\n=== Trade Statistics ===")
print(f"Total Trades:           {len(trade_returns)}")
print(f"Win Rate:               {win_rate:.2%}")
print(f"Average Win:            {avg_win:.2%}")
print(f"Average Loss:           {avg_loss:.2%}")
print(f"Profit Factor:          {profit_factor:.2f}")
print(f"Expectancy:             {expectancy:.2%}")
print("\n=== Directional Accuracy ===")
print(f"Signal Accuracy:        {acc_dir:.2%}")
print("Confusion Matrix (−1,0,+1):")
print(cm)
print("\nClassification Report:")
print(clf_report)

# -----------------------------------------------------------------------------
# 7) Plots
# -----------------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(equity_curve, label="Equity")
plt.title("Equity Curve")
plt.xlabel("Trade Number")
plt.ylabel("Account Value")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10,4))
plt.plot(drawdown, color="red", label="Drawdown")
plt.title("Drawdown Curve")
plt.xlabel("Trade Number")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,4))
plt.hist(trade_returns, bins=50)
plt.title("Distribution of Trade Returns")
plt.xlabel("Return per Trade")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()
