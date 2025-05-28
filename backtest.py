import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from quantization import Quantization
from train import model   # your pretrained model

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CSV_PATH        = "historical_data/XAUUSD_1h_historical_data.csv"
N_DATAPOINTS    = 500
N_BITS          = 50
WINDOW          = 3
TRAIN_RATIO     = 0

INITIAL_CAP     = 5_000.0
LEVERAGE        = 100.0
RISK_PER_TRADE  = 0.30
STOP_LOSS_PCT   = 0.002   # 0.5%
TAKE_PROFIT_PCT = 0.002   # 1.0%

# -----------------------------------------------------------------------------
# 1) Load & preprocess CSV directly
# -----------------------------------------------------------------------------
df = (
    pd.read_csv(CSV_PATH)
      .filter(["open","high","low","close","tick_volume"])
      .rename(columns={
          "open": "Open",
          "high": "High",
          "low": "Low",
          "close": "Close",
          "tick_volume": "Volume"
      })
)
df = df.iloc[-1*N_DATAPOINTS:].reset_index(drop=True)

closes = df["Close"].values
highs  = df["High"].values
lows   = df["Low"].values

# -----------------------------------------------------------------------------
# 2) Quantize & one‐hot encode entire series
# -----------------------------------------------------------------------------
quant     = Quantization(n_bits=N_BITS)
labels    = quant.fit_transform(closes)
one_hot   = np.eye(N_BITS, dtype=int)[labels]

# compute split index (first test bar)
total_samples = len(one_hot) - WINDOW
split_idx     = int(total_samples * TRAIN_RATIO) + WINDOW

# -----------------------------------------------------------------------------
# 3) Real‐time backtest loop with TP & SL
# -----------------------------------------------------------------------------
capital       = INITIAL_CAP
equity_curve  = [capital]
trade_returns = []
signals_list  = []
pred_prev     = None

for t in range(split_idx, len(one_hot) - 1):
    # prepare model input
    window_seq = one_hot[t-WINDOW : t]
    X_input    = window_seq[np.newaxis, :, :]

    # model prediction
    y_prob   = model.predict(X_input, verbose=0)[0]
    pred_lbl = int(np.argmax(y_prob))

    # generate signal
    signal = 0
    if pred_prev is not None:
        if   pred_lbl > pred_prev:
            signal = +1
        elif pred_lbl < pred_prev:
            signal = -1
    signals_list.append(signal)
    pred_prev = pred_lbl

    # execute trade if signal
    if signal != 0:
        entry_price = closes[t]
        # TP/SL levels
        if signal == +1:
            tp = entry_price * (1 + TAKE_PROFIT_PCT)
            sl = entry_price * (1 - STOP_LOSS_PCT)
        else:
            tp = entry_price * (1 - TAKE_PROFIT_PCT)
            sl = entry_price * (1 + STOP_LOSS_PCT)

        hi, lo, next_close = highs[t+1], lows[t+1], closes[t+1]

        # exit logic: TP first, then SL, else close
        if   signal == +1 and hi  >= tp:        exit_price = tp
        elif signal == -1 and lo  <= tp:        exit_price = tp
        elif signal == +1 and lo  <= sl:        exit_price = sl
        elif signal == -1 and hi  >= sl:        exit_price = sl
        else:                                   exit_price = next_close

        # PnL calculation
        margin   = capital * RISK_PER_TRADE
        notional = margin * LEVERAGE
        units    = notional / entry_price
        pnl      = units * ((exit_price - entry_price) if signal==+1
                            else (entry_price - exit_price))

        capital += pnl
        trade_returns.append(pnl / margin)

    equity_curve.append(capital)

equity_curve  = np.array(equity_curve)
trade_returns = np.array(trade_returns)

# -----------------------------------------------------------------------------
# 4) Metrics & Analytics
# -----------------------------------------------------------------------------
test_labels = labels[split_idx : len(one_hot)]
true_dirs   = np.sign(np.diff(test_labels))
pred_dirs   = np.array(signals_list)

cm         = confusion_matrix(true_dirs, pred_dirs, labels=[-1,0,1])
clf_report = classification_report(true_dirs, pred_dirs, labels=[-1,0,1])

n_trades   = len(trade_returns)
cum_return = equity_curve[-1]/INITIAL_CAP - 1
hours      = len(equity_curve)
years      = hours/(24*365)
ann_return = (1+cum_return)**(1/years) - 1

vol      = np.std(trade_returns)*np.sqrt(n_trades/years) if n_trades else np.nan
down_std = np.std(trade_returns[trade_returns<0])*np.sqrt(n_trades/years) if n_trades else np.nan
sharpe   = ann_return/vol      if vol      else np.nan
sortino  = ann_return/down_std if down_std else np.nan

peak     = np.maximum.accumulate(equity_curve)
drawdown = equity_curve/peak - 1
max_dd   = drawdown.min()

# -----------------------------------------------------------------------------
# 5) Reporting
# -----------------------------------------------------------------------------
print("\n=== Performance Summary ===")
print(f"Cumulative Return:      {cum_return:.2%}")
print(f"Annualized Return:      {ann_return:.2%}")
print(f"Sharpe Ratio:           {sharpe:.2f}")
print(f"Sortino Ratio:          {sortino:.2f}")
print(f"Max Drawdown:           {max_dd:.2%}\n")

print("=== Directional Accuracy ===")
print("Confusion Matrix (−1,0,+1):")
print(cm)
print("\nClassification Report:")
print(clf_report)

# -----------------------------------------------------------------------------
# 6) Plots
# -----------------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(equity_curve, label="Equity Curve")
plt.title("Equity Curve")
plt.xlabel("Step")
plt.ylabel("Account Value")
plt.grid()
plt.legend()

plt.figure(figsize=(10,4))
plt.plot(drawdown, color="red", label="Drawdown")
plt.title("Drawdown Curve")
plt.xlabel("Step")
plt.ylabel("Drawdown")
plt.grid()
plt.legend()

plt.figure(figsize=(8,4))
plt.hist(trade_returns, bins=50)
plt.title("Distribution of Trade Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.grid()

plt.show()
