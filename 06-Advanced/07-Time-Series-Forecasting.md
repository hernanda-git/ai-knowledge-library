# Time-Series and Forecasting with Machine Learning
## Table of Contents
1. [Introduction](#1-introduction)
2. [Classical Methods](#2-classical)
   2.1 [ARIMA](#21-arima)
   2.2 [Exponential Smoothing](#22-exponential-smoothing)
   2.3 [Prophet](#23-prophet)
   2.4 [Evaluation Metrics](#24-evaluation-metrics)
3. [ML for Time Series](#3-ml)
   3.1 [Feature Engineering](#31-feature-engineering)
   3.2 [Models](#32-models)
4. [Deep Learning](#4-dl)
   4.1 [Architectures](#41-architectures)
   4.2 [Multi-Horizon Forecasting](#42-multi-horizon-forecasting)
   4.3 [LSTM Time-Series Example](#43-lstm-time-series-example)
   4.4 [DeepAR](#44-deepar)
5. [Probabilistic Forecasting](#5-probabilistic)
6. [Anomaly Detection](#6-anomaly)
7. [Transformer-Based TS Models](#7-transformer-based-time-series-models)
8. [Time Series Cross-Validation](#8-time-series-cross-validation)
9. [Multi-Step Forecasting Strategies](#9-multi-step-forecasting-strategies)
10. [Causal & Structural TS Models](#10-causal-and-structural-time-series-models)
11. [Deployment Considerations](#11-deployment-considerations-for-forecasting-pipelines)
11a. [Ensemble Methods for Time Series](#11a-ensemble-methods-for-time-series)
11b. [Financial and Economic Time Series Applications](#11b-financial-and-economic-time-series-applications)
12. [Cross-References](#12-cross-references)
---
## 1. Introduction
Time-series data is everywhere: stock prices, sensor readings, web traffic, energy consumption, weather, medical vitals. Forecasting — predicting future values — is one of the most valuable ML applications with direct business impact (inventory optimization, demand planning, predictive maintenance).
### Key Characteristics
- **Temporal dependence:** Observations are not independent
- **Trend:** Long-term direction (increasing, decreasing, stable)
- **Seasonality:** Regular patterns (daily, weekly, yearly)
- **Cycles:** Irregular longer-term patterns
- **Noise:** Random variation
- **Stationarity:** Statistical properties constant over time (or not)
---
## 2. Classical Methods
### 2.1 ARIMA (AutoRegressive Integrated Moving Average)
ARIMA(p,d,q): AR(p) + I(d) + MA(q)
- AR(p): use p past values as predictors
- I(d): difference d times to achieve stationarity
- MA(q): use q past forecast errors as predictors
**Limitations:** Linear only, univariate, assumes stationarity after differencing

#### Stationarity Testing (ADF & KPSS)
Verify stationarity before fitting ARIMA:
| Test | Null Hypothesis (H0) | Stationary if |
|------|----------------------|:-------------:|
| **ADF** | Series has unit root (non-stationary) | p-value < 0.05 |
| **KPSS** | Series is stationary | p-value >= 0.05 |
**Tip:** Use both — ADF may fail on near-stationary data; KPSS may reject too easily on long series. Cross-check with visual inspection.

```python
from statsmodels.tsa.stattools import adfuller, kpss
def stationarity_test(series, sig=0.05):
    adf_p = adfuller(series.dropna(), autolag='AIC')[1]
    kpss_p = kpss(series.dropna(), regression='c', nlags='auto')[1]
    adf_s, kpss_s = adf_p < sig, kpss_p >= sig
    print(f"ADF p={adf_p:.5f} -> {'Stationary' if adf_s else 'Non-stationary'}")
    print(f"KPSS p={kpss_p:.5f} -> {'Stationary' if kpss_s else 'Non-stationary'}")
    if adf_s and kpss_s: print("=> Strong evidence of stationarity")
    elif adf_s != kpss_s: print("=> Mixed signals - check for trend/slow decay")
    else: print("=> Differencing needed")
    return adf_s and kpss_s
```

#### ARIMA Modeling with statsmodels

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
                 header=0, index_col=0, parse_dates=True)
series = np.log(df['Passengers'])  # stabilize variance
diff = series.diff().dropna()
plot_acf(diff, lags=20); plot_pacf(diff, lags=20); plt.tight_layout(); plt.show()

fitted = ARIMA(series, order=(1, 1, 1)).fit()  # ARIMA(1,1,1) from ACF/PACF
print(fitted.summary())
forecast = np.exp(fitted.forecast(steps=12))
plot_acf(fitted.resid, lags=20, title='Residual ACF (white noise check)'); plt.show()
```
**Tips:** Use `pmdarima.auto_arima` for automatic selection. Compare AIC/BIC. Check residuals for autocorrelation (Ljung-Box test). Avoid high p/q on short series.
---
### 2.2 Exponential Smoothing
Weighted average of past observations with exponentially decaying weights.
- **Simple (SES):** single parameter, no trend/seasonality
- **Holt's:** add trend component
- **Holt-Winters:** add seasonality component (additive or multiplicative)
- Damped trend: reduce trend over long forecast horizons
---
### 2.3 Prophet (Facebook, 2017)
Decomposable model: y(t) = g(t) + s(t) + h(t) + epsilon(t)
- g(t): trend (piecewise linear or logistic growth with changepoints)
- s(t): seasonality (Fourier series)
- h(t): holiday effects (user-specified)
- Robust to missing data, handles outliers, automatic changepoint detection

#### Prophet Code Example

```python
import pandas as pd, numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
                 header=0, names=['ds', 'y'], parse_dates=['ds'])
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode='multiplicative', changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(df)
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)
model.plot(forecast); plt.show()
model.plot_components(forecast); plt.show()
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
```
**Tips:** Use `seasonality_mode='multiplicative'` when seasonal amplitude grows with trend. Higher `changepoint_prior_scale` -> more flexible trend. Set `uncertainty_samples=0` for faster fitting. Cross-validate with `prophet.diagnostics.cross_validation()`.
---
### 2.4 Evaluation Metrics for Time-Series Forecasting

| Metric | Formula | Range | Interpretation |
|--------|---------|:-----:|----------------|
| **RMSE** | sqrt(mean((y - y_hat)^2)) | [0, inf) | Penalizes large errors; same units as target |
| **MAE** | mean(\|y - y_hat\|) | [0, inf) | Average absolute error; directly interpretable |
| **MAPE** | 100 * mean(\|(y - y_hat)/y\|) | [0, inf] | Percentage error; undefined if y=0 |
| **sMAPE** | 200 * mean(\|y - y_hat\|/(\|y\|+\|y_hat\|)) | [0, 200] | Symmetric; bounds extremes; used in M4/M5 |
| **MASE** | MAE / MAE(naive) | [0, inf) | < 1.0 means better than naive forecast |

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y, p): return np.mean(np.abs((y - p) / y)) * 100
def smape(y, p):
    d = np.abs(y) + np.abs(p); d[d == 0] = 1e-10
    return np.mean(2.0 * np.abs(y - p) / d) * 100
def mase(y, p, y_train):
    nm = np.mean(np.abs(np.diff(y_train.flatten())))
    return mean_absolute_error(y, p) / nm if nm else np.nan

y, p, t = np.array([100, 120, 110, 130]), np.array([102, 118, 112, 127]), np.array([90, 95, 100, 105, 110])
print(f"RMSE={mean_squared_error(y, p, squared=False):.3f} MAE={mean_absolute_error(y, p):.3f}")
print(f"MAPE={mape(y, p):.3f}% sMAPE={smape(y, p):.3f}% MASE={mase(y, p, t):.3f}")
```
**When to use which:** RMSE/MAE for consistent scale; MAPE for business (% error intuitive - fails at zero); sMAPE for symmetric percentage (competition standard); MASE for cross-series comparison (scale-independent, zero-safe).
---
## 3. ML for Time Series
### 3.1 Feature Engineering for ML Forecasters
Convert time-series to supervised learning: use past lags as features.

**Features:**
- **Lagged values:** y_{t-1}, y_{t-2}, ..., y_{t-7} (daily)
- **Calendar features:** hour, day of week, month, holiday, quarter, fiscal period
- **Rolling statistics:** rolling mean, std, min, max (7-day, 14-day, 30-day)
- **Seasonal features:** distance to holiday, days since last event
- **Exogenous:** weather, price, promotion, competitor activity
- **Time since:** last outlier, last event, last data change
- **Fourier features:** sin(2*pi*f*t), cos(2*pi*f*t) for known periods

#### Code: Creating Lag Features and Rolling Statistics

```python
import pandas as pd, numpy as np

def create_lag_features(df, target_col, lags=None, rolling_windows=None):
    if lags is None: lags = [1, 2, 3, 7, 14, 28]
    if rolling_windows is None: rolling_windows = [7, 14, 30]
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    for w in rolling_windows:
        s = df[target_col].shift(1)
        df[f'{target_col}_rolling_mean_{w}'] = s.rolling(w).mean()
        df[f'{target_col}_rolling_std_{w}'] = s.rolling(w).std()
        df[f'{target_col}_rolling_min_{w}'] = s.rolling(w).min()
        df[f'{target_col}_rolling_max_{w}'] = s.rolling(w).max()
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None
    if idx is not None:
        df[['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']] = (
            idx.hour, idx.dayofweek, idx.quarter, idx.month, idx.year, idx.dayofyear)
        df['weekend'] = (idx.dayofweek >= 5).astype(int)
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_7'] = df[target_col].diff(7)
    return df

dates = pd.date_range('2024-01-01', periods=100, freq='D')
sales = 100 + np.cumsum(np.random.randn(100)) + 10 * np.sin(2 * np.pi * np.arange(100) / 7)
features = create_lag_features(pd.DataFrame({'sales': sales}, index=dates), 'sales', lags=[1, 2, 7], rolling_windows=[7])
print(features.head(10))
```
**Caution:** Shift creates NaN in first `max(lag)` rows — drop or impute before training.
---
### 3.2 Models
| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **XGBoost/LightGBM** | Great with features, handles non-linearity | Doesn't extrapolate trend | Demand forecasting, classification |
| **Random Forest** | Robust, interpretable | Poor extrapolation | Short-term forecasting |
| **Linear Regression** | Simple, interpretable | Underfits | Baseline |
| **Lasso/Ridge** | Feature selection | Linear only | High-dimensional features |
---
## 4. Deep Learning for Time Series
### 4.1 Architectures
| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **LSTM** | Long-range dependencies, flexible | Slow training, less interpretable |
| **GRU** | Faster than LSTM, similar accuracy | Same limitations as LSTM |
| **CNN (TCN)** | Parallel training, stable gradients | Limited receptive field (dilated conv helps) |
| **Transformer (Informer, PatchTST, TimesNet)** | Multi-horizon, interpretable attention | Computationally expensive |
| **N-BEATS** | Pure MLP, interpretable via basis decomposition | Univariate only |
| **Temporal Fusion Transformer (TFT)** | Interpretable attention + feature importance | Complex training |
### 4.2 Multi-Horizon Forecasting
Predict multiple steps ahead simultaneously:
- **Direct:** One model per horizon (n models for n steps)
- **Recursive:** Use prediction as input for next (compounds error)
- **Seq2Seq:** Encoder-decoder architecture (standard deep learning approach)
- **N-BEATS/TFT:** SOTA multi-horizon models
### 4.3 LSTM Time-Series Example

**PyTorch Implementation**
```python
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42); torch.manual_seed(42)
t = np.arange(0, 500)
data = 10 + 0.05*t + 5*np.sin(2*np.pi*t/20) + np.random.normal(0, 1, t.shape)

def seqs(data, sl=20):
    X, y = [], []
    for i in range(len(data)-sl):
        X.append(data[i:i+sl]); y.append(data[i+sl])
    return np.array(X).reshape(-1, sl, 1), np.array(y).reshape(-1, 1)

X, y = seqs(data)
split = int(0.8*len(X))
X_train, y_train = torch.FloatTensor(X[:split]), torch.FloatTensor(y[:split])
X_test, y_test = torch.FloatTensor(X[split:]), torch.FloatTensor(y[split:])
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

class LSTMFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1, :])

model = LSTMFC(); opt = torch.optim.Adam(model.parameters(), lr=0.001); loss_fn = nn.MSELoss()
for epoch in range(100):
    for bx, by in loader:
        opt.zero_grad(); loss_fn(model(bx), by).backward(); opt.step()
    if (epoch+1) % 25 == 0:
        print(f"Epoch {epoch+1}: loss={loss_fn(model(bx), by):.6f}")
model.eval()
with torch.no_grad():
    print(f"Test RMSE: {torch.sqrt(loss_fn(model(X_test), y_test)):.3f}")
```
**Keras/TensorFlow Implementation**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(20, 1)), Dropout(0.2),
    LSTM(64, return_sequences=False), Dropout(0.2), Dense(1)
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[es], verbose=0)
print(f"Test loss: {model.evaluate(X_test, y_test):.6f}")
```
**LSTM Tips:** Scale data before feeding (MinMaxScaler/StandardScaler). Use `return_sequences=True` in intermediate layers. For multi-step forecasting, use encoder-decoder architecture. Start with 1-2 layers.
---
### 4.4 DeepAR (Amazon, 2020)
Autoregressive RNN with probabilistic output:
1. Encoder RNN processes past window
2. Decoder RNN generates future distribution one step at a time
3. Each step: sample from distribution, pass as next input (during training: ground truth)
4. Output: full predictive distribution
---
## 5. Probabilistic Forecasting
Instead of point forecasts, predict the full distribution.
| Method | Output | Complexity |
|--------|--------|:----------:|
| **Quantile regression** | Predict specific quantiles (p10, p50, p90) | Low |
| **Conformal prediction** | Prediction intervals with coverage guarantees | Low |
| **Bayesian methods** | Full posterior distribution | High |
| **DeepAR/TFT** | Likelihood parameters (mu, sigma) | Moderate |
| **Diffusion/TTA (Time Diffusion)** | Generative probabilistic forecasting | Very high |
---
## 6. Anomaly Detection
### 6.1 Types
- **Point anomalies:** Individual data points that deviate (transaction fraud)
- **Contextual anomalies:** Unusual in context (35 deg C in winter = anomaly, in summer = normal)
- **Collective anomalies:** Unusual sequence (DDOS attack traffic pattern)
### 6.2 Methods
| Method | Unsupervised? | Best For |
|--------|:------------:|----------|
| **IQR (Interquartile Range)** | Yes | Fast, simple point anomalies |
| **Isolation Forest** | Yes | Multi-dimensional, fast |
| **Autoencoder** | Yes | Reconstruction error for complex patterns |
| **LSTM Autoencoder** | Yes | Sequential data, pattern anomalies |
| **SPOT/EVT** | Yes | Extreme value detection |
| **SR-CNN** | Yes | Spectral residual for time series |
|---
## 7. Transformer-Based Time Series Models

Transformer architectures have revolutionized time-series forecasting by effectively capturing long-range dependencies and complex temporal patterns. Three notable models — PatchTST, TimesNet, and Informer — each bring unique innovations.

| Model | Year | Key Innovation | Architecture | Strengths | Limitations |
|-------|------|---------------|-------------|-----------|-------------|
| **Informer** | 2021 | ProbSparse self-attention, self-attention distilling | Encoder-decoder Transformer | Handles long sequences efficiently; O(L log L) complexity; generative-style decoder for long-horizon | Requires careful hyperparameter tuning; less effective on short sequences |
| **PatchTST** | 2023 | Patching time series into subseries-level patches | Channel-independent patch encoder | Reduces computational cost; preserves local semantics; strong on multivariate with CI strategy | Channel independence loses inter-variable correlations; requires fixed patch length |
| **TimesNet** | 2023 | 2D convolution of 1D time series via FFT periodicity | Inception-style 2D CNN blocks | Captures both intra-period and inter-period variations; robust across frequencies | Periodic assumption may fail on aperiodic data; higher GPU memory usage |

### Code: Simplified ProbSparse Self-Attention Sketch

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

def prob_sparse_attention(Q, K, V, n_top=25):
    """Simplified ProbSparse: top-k queries from sparsity measurement."""
    B, H, L, d = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    # Sparsity measure: M(q_i, K) = max_j(q_i*k_j^T) - mean_j(q_i*k_j^T)
    m = scores.max(dim=-1).values - scores.mean(dim=-1)    # (B, H, L)
    top_idx = m.topk(n_top, dim=-1).indices                 # (B, H, n_top)
    Q_sel = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, d))
    attn = F.softmax(torch.matmul(Q_sel, K.transpose(-2, -1)) / math.sqrt(d), dim=-1)
    out = torch.matmul(attn, V)                            # (B, H, n_top, d)
    # Scatter back and average for demonstration
    return out.mean(dim=1).mean(dim=1)
```

**When to use which:** Informer for very long sequences (>1000 steps), PatchTST for multivariate data where local subseries patterns matter, TimesNet for data with strong multi-periodicity (e.g., energy with daily + weekly + yearly cycles).

---

## 8. Time Series Cross-Validation

Standard k-fold CV leaks temporal order. Use specialized strategies that respect the time axis.

### 8.1 Expanding Window (Anchored CV)
Train set grows forward; test set is a fixed window ahead. Mimics real-world retraining.

```python
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error

def expanding_window_cv(data, n_splits=5, test_size=12):
    """Expanding window cross-validation for time series.
    Train: fixed start, expanding end. Test: fixed-size window after train.
    """
    n = len(data)
    scores = []
    for i in range(n_splits):
        train_end = (i + 1) * (n - test_size * n_splits) // n_splits + test_size * i
        train = data.iloc[:train_end]
        test = data.iloc[train_end:train_end + test_size]
        if len(test) < test_size:
            break
        pred = np.full(len(test), train.iloc[-1].values[0])  # naive: last value
        score = mean_squared_error(test, pred, squared=False)
        scores.append(score)
        print(f"Fold {i+1}: train=0:{train_end}, test={train_end}:{train_end+test_size}, RMSE={score:.3f}")
    return np.mean(scores), scores

# Example usage
dates = pd.date_range('2020-01-01', periods=120, freq='M')
series = pd.DataFrame(np.cumsum(np.random.randn(120)) + 50, index=dates, columns=['value'])
avg_rmse, fold_scores = expanding_window_cv(series, n_splits=4, test_size=6)
print(f"Mean RMSE: {avg_rmse:.3f}")
```

### 8.2 Sliding Window (Rolling CV)
Both train and test windows advance — always using the same training window size.

```python
def sliding_window_cv(data, window_size=60, test_size=12, step=12):
    """Sliding window CV: fixed train window slides forward."""
    scores = []
    start = 0
    while start + window_size + test_size <= len(data):
        train = data.iloc[start:start + window_size]
        test = data.iloc[start + window_size:start + window_size + test_size]
        pred = np.full(len(test), train.iloc[-1].values[0])
        score = mean_squared_error(test, pred, squared=False)
        scores.append(score)
        print(f"Fold: train={start}:{start+window_size}, test={start+window_size}:{start+window_size+test_size}, RMSE={score:.3f}")
        start += step
    return np.mean(scores), scores

avg_rmse_s, fold_scores_s = sliding_window_cv(series, window_size=60, test_size=6, step=12)
print(f"Mean RMSE (sliding): {avg_rmse_s:.3f}")
```

| Strategy | Train Size | Bias-Variance | Best For |
|----------|-----------|:------------:|----------|
| **Expanding Window** | Grows over folds | More data → lower variance | Long datasets, retraining at fixed intervals |
| **Sliding Window** | Fixed | Stable bias; may miss old patterns | Non-stationary data, concept drift |
| **Walk-Forward (Purged)** | Expanding + gap before test | Avoids data leakage | Financial time series (avoid look-ahead) |

> **Caution:** Always purge any observation within `max(lag)` of the test set from the training set to prevent leakage. For financial data, add a gap (e.g., one quarter) between train and test.

---

## 9. Multi-Step Forecasting Strategies — Deeper Dive

The overview in §4.2 introduced the strategies; here we compare them in depth with code and trade-offs.

### 9.1 Direct Strategy

Train `H` separate models, one for each forecast horizon. Models are independent — errors don't compound.

| Horizon | Features | Target |
|---------|----------|--------|
| h=1     | y_{t-1}, ..., y_{t-p} | y_{t+1} |
| h=2     | y_{t-1}, ..., y_{t-p} | y_{t+2} |
| ...     | ... | ... |
| h=H     | y_{t-1}, ..., y_{t-p} | y_{t+H} |

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd, numpy as np

def direct_forecast(train, horizon=12, lags=[1, 2, 3, 7, 14]):
    """Train one model per horizon."""
    models = {}
    for h in range(1, horizon + 1):
        X, y = [], []
        for i in range(max(lags), len(train) - h):
            X.append([train.iloc[i - lag] for lag in lags])
            y.append(train.iloc[i + h])
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)
        mdl.fit(np.array(X), np.array(y))
        models[h] = mdl
    last_features = np.array([train.iloc[-lag] for lag in lags]).reshape(1, -1)
    forecasts = [models[h].predict(last_features)[0] for h in range(1, horizon + 1)]
    return np.array(forecasts)

train = np.cumsum(np.random.randn(200)) + 50
preds_direct = direct_forecast(train, horizon=12)
print("Direct forecasts:", preds_direct.round(2))
```

**Pros:** No error compounding; each horizon can use a different algorithm. **Cons:** `H` models to train and maintain; ignores inter-horizon dependencies.

### 9.2 Recursive (Iterated) Strategy

Train a single one-step model and feed predictions back as input.

```python
def recursive_forecast(train, horizon=12, lags=[1, 2, 3, 7, 14]):
    """Single one-step model, applied iteratively."""
    X, y = [], []
    for i in range(max(lags), len(train) - 1):
        X.append([train.iloc[i - lag] for lag in lags])
        y.append(train.iloc[i + 1])
    mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(np.array(X), np.array(y))
    history = list(train)
    forecasts = []
    for _ in range(horizon):
        features = np.array([history[-lag] for lag in lags]).reshape(1, -1)
        pred = mdl.predict(features)[0]
        forecasts.append(pred)
        history.append(pred)
    return np.array(forecasts)

preds_recursive = recursive_forecast(train, horizon=12)
print("Recursive forecasts:", preds_recursive.round(2))
```

**Pros:** Single model, easy to maintain; captures temporal continuity. **Cons:** Error compounds over horizons; sensitive to distribution drift.

### 9.3 Seq2Seq Strategy

Encoder-decoder architecture that directly outputs the full horizon.

```python
import torch, torch.nn as nn

class Seq2SeqTS(nn.Module):
    """Minimal sequence-to-sequence for multi-step forecasting."""
    def __init__(self, input_len=20, hidden=64, output_len=12):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden, batch_first=True)
        self.decoder = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h, c) = self.encoder(x)
        decoder_input = x[:, -1:, :]
        outputs = []
        for _ in range(self.decoder_output_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred
        return torch.cat(outputs, dim=1)

Seq2SeqTS.decoder_output_len = 12
model = Seq2SeqTS()
dummy = torch.randn(4, 20, 1)
out = model(dummy)
print(f"Seq2Seq output shape: {out.shape}")
```

**Pros:** End-to-end multi-step; learns temporal dependencies jointly. **Cons:** Complex training (teacher forcing, scheduled sampling); sensitive to input length.

### 9.4 Strategy Comparison

| Strategy | Models | Error Propagation | Comput. Cost | Maintenance | Best When |
|----------|:-----:|:----------------:|:-----------:|:-----------:|-----------|
| **Direct** | H | None | High (H models) | High | Horizons are independent |
| **Recursive** | 1 | Compounds | Low | Low | Short horizons, stable dynamics |
| **Seq2Seq** | 1 | Moderate | Medium | Medium | Complex patterns, long horizons |
| **DIRMO** | 1 per block | Reduces compounding | Medium | Medium | Compromise between direct & recursive |

> **DIRMO (Direct-Recursive Mixture):** Group horizons into blocks, train one model per block, use recursion within each block. Balances error compounding and model count.

---

## 10. Causal and Structural Time Series Models

When forecasting needs to answer *why* a change happened, or when external interventions affect the system, causal and structural approaches go beyond pure pattern matching.

### 10.1 Structural Time Series Models (STS)

Explicitly decompose the series into interpretable components:

> y(t) = trend(t) + seasonality(t) + cycle(t) + regression(t) + error(t)

| Implementation | Framework | Key Features |
|---------------|-----------|-------------|
| **UnobservedComponents** | statsmodels | MLE estimation, local linear trend, seasonal, cycle |
| **TensorFlow Probability STS** | TFP | Bayesian structural models, variational inference |
| **Prophet** | Meta | De facto STS for business (see §2.3) |
| **BSTS** | R (`bsts`) | Spike-and-slab priors for feature selection |

```python
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

t = np.arange(200)
data = 0.1 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.5, 200)
model = UnobservedComponents(data, level='local linear trend', seasonal=12)
result = model.fit(disp=False)
print(f"STS AIC: {result.aic:.1f}, Forecast:\n{result.forecast(steps=12).round(2)}")
```

### 10.2 Causal Impact (Google, 2014)

Estimates the effect of an intervention (marketing campaign, policy change) by constructing a synthetic counterfactual from unaffected control series.

- **Framework:** BSTS model predicts what *would have happened* without the intervention.
- **Causal effect:** actual − counterfactual (with Bayesian uncertainty intervals).
- **Available:** `CausalImpact` (R), `tfcausalimpact` (Python), or custom diff-in-diff with Prophet.

### 10.3 Granger Causality

Statistical test: does lagged `X` help predict `Y` beyond lagged `Y` alone?

```python
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np, pandas as pd

df = pd.DataFrame({
    'sales': np.cumsum(np.random.randn(200)) + 50,
    'spend': np.cumsum(np.random.randn(200)) + 20
})
# Test if 'spend' Granger-causes 'sales' (up to 4 lags)
gc = grangercausalitytests(df[['sales', 'spend']], maxlag=4, verbose=True)
# Low p-value (< 0.05) → 'spend' Granger-causes 'sales'
```

> **Note:** Granger causality tests *predictive* precedence, not true causal mechanisms. Use DAGs or instrumental variables for causal claims.

### 10.4 When to Use Causal/Structural Models

| Scenario | Recommended Approach |
|----------|---------------------|
| Decomposing trend + seasonality | STS (UnobservedComponents) |
| Estimating intervention effect | Causal Impact (BSTS with controls) |
| Testing variable precedence | Granger causality |
| Policy/regulatory forecasting | Structural econometric models (VAR, SVAR) |
| Feature selection + forecasting | BSTS with spike-and-slab |

---

## 11. Deployment Considerations for Forecasting Pipelines

Moving from notebooks to production forecasting requires addressing timing, staleness, and operational concerns unique to time series.

### 11.1 Pipeline Architecture

```
[Data Ingestion] → [Validation] → [Feature Engineering] → [Model Inference] → [Post-processing] → [Storage/Serving]
       ↕                    ↕                  ↕                     ↕
  [Backfill/Retrain]  [Anomaly Alert]   [Lag Alignment]     [Ensemble/Blend]
```

### 11.2 Key Production Concerns

| Concern | Description | Mitigation |
|---------|-------------|------------|
| **Feature Lag Alignment** | Latest features may not be available for all lags at inference time | Maintain a lag budget; pad with last-known values; use lag-1 of all features for real-time |
| **Model Staleness** | Forecast accuracy degrades as data distribution shifts | Monitor drift (PSI, KS-test); auto-retrain triggers; champion/challenger evaluation |
| **Inference Latency** | Batch vs real-time vs scheduled forecasts | Batch for 1000+ series; on-demand for single series; scheduled for daily/hourly pipelines |
| **Backtesting Integrity** | Train/test leakage from improper CV | Purging (gap between train/test); temporal ordering enforcement; dated snapshots |
| **Data Freshness** | Partial-day data; irregular sampling | Missing value imputers; as-of-time joins; versioned feature tables |
| **Forecast Reconciliation** | Hierarchical forecasts (top-down, bottom-up, optimal combination) | MinT (Minimum Trace) reconciliation; budget constraints per level |

### 11.3 Production Code Pattern

```python
import logging, datetime, json
from typing import Dict, Optional
import pandas as pd, numpy as np
from dataclasses import dataclass

@dataclass
class ForecastConfig:
    model_path: str = '/models/forecaster.pkl'
    horizon: int = 12
    frequency: str = 'D'
    retrain_days: int = 7
    drift_threshold: float = 0.15
    min_training_points: int = 60

class ForecastingPipeline:
    """Production forecasting pipeline with drift detection and retraining."""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        import joblib
        try:
            self.model = joblib.load(self.config.model_path)
        except FileNotFoundError:
            logging.warning("No model found — will train on first run")

    def predict(self, series: pd.Series) -> Dict:
        if len(series) < self.config.min_training_points:
            raise ValueError(f"Need ≥{self.config.min_training_points} points")
        self._check_drift(series)
        forecast = self._run_inference(series)
        return {"forecast": forecast.tolist(), "timestamp": datetime.datetime.utcnow().isoformat()}

    def _check_drift(self, series: pd.Series) -> bool:
        if self.model is None:
            return False
        recent = series.iloc[-30:].values
        train_ref = getattr(self.model, 'train_stats_', {}).get('mean', 0)
        drift = float(np.abs(np.mean(recent) - train_ref) / (train_ref + 1e-8))
        if drift > self.config.drift_threshold:
            logging.warning(f"Drift detected: {drift:.3f}")
        return drift > self.config.drift_threshold

    def _run_inference(self, series):
        return np.full(self.config.horizon, series.iloc[-1])

config = ForecastConfig(model_path='/models/demand_forecaster.pkl', horizon=12)
pipeline = ForecastingPipeline(config)
series = pd.Series(np.cumsum(np.random.randn(200)) + 100)
result = pipeline.predict(series)
print(json.dumps(result, indent=2))
```

### 11.4 Monitoring and Alerts

| Metric | What It Detects | Alert Threshold |
|--------|----------------|:---------------:|
| **Forecast Error (RMSE/MAE)** | Model accuracy degradation | > 1.5× baseline error |
| **Prediction Interval Width** | Model uncertainty increase | > 2× historical mean width |
| **Feature Drift (PSI)** | Input distribution shift | > 0.2 PSI |
| **Data Freshness Lag** | Delayed data arrival | > 1 scheduled interval |
| **Model Age** | Time since last retrain | > 2× retrain_days |
| **Null/NaN Forecasts** | Pipeline failure | Any null output |

### 11.5 Deployment Checklist

- [ ] Timestamps are validated and sorted before training
- [ ] Lag features are computed with `shift()` and NaN rows dropped
- [ ] Evaluation uses temporal CV (not random shuffle)
- [ ] Model artifacts include training date and feature schema
- [ ] Forecast horizon is aligned with business decision cadence
- [ ] Retraining is automated with drift detection triggers
- [ ] Fallback model (e.g., naive seasonal) exists for pipeline failures
- [ ] Hierarchical forecasts are reconciled if multiple aggregation levels needed
- [ ] Prediction intervals / uncertainty estimates are served alongside point forecasts
- [ ] Champion/challenger A/B testing framework compares model versions

---

## 11a. Ensemble Methods for Time Series

Combining multiple forecasting models often yields more accurate and robust predictions than any single model. Ensemble methods are especially valuable in time series where model uncertainty is high and no single approach dominates across all regimes.

### 11a.1 Why Ensemble for Time Series?

| Reason | Explanation |
|--------|-------------|
| **Bias-variance trade-off** | Averaging reduces variance without increasing bias |
| **Model complementarity** | Different models capture different patterns (linear vs non-linear, short vs long-term) |
| **Regime robustness** | One model may excel in stable periods, another during volatility |
| **Uncertainty quantification** | Ensemble spread provides natural prediction intervals |
| **Competition-winning** | M4/M5 forecasting competitions are won by ensembles, not single models |

### 11a.2 Ensemble Strategies

**Weighted Averaging:**
```python
def weighted_ensemble(forecasts, weights=None):
    """Weighted average of multiple forecasts.
    
    Args:
        forecasts: list of numpy arrays, each shape (horizon,)
        weights: optional array of weights (sums to 1). If None, uses equal weights.
    """
    if weights is None:
        weights = np.ones(len(forecasts)) / len(forecasts)
    return np.sum(np.array(forecasts).T * np.array(weights), axis=1)

# Example: combine ARIMA, Prophet, and LSTM
arima_fcst = np.array([102, 105, 108, 110])
prophet_fcst = np.array([101, 106, 107, 112])
lstm_fcst = np.array([103, 104, 109, 111])

# Equal weights
equal = weighted_ensemble([arima_fcst, prophet_fcst, lstm_fcst])
print(f"Equal-weighted ensemble: {equal.round(2)}")

# Performance-based weights (inverse error weighting)
errors = np.array([0.12, 0.15, 0.10])  # validation RMSEs
weights = (1/errors) / (1/errors).sum()
perf_weighted = weighted_ensemble([arima_fcst, prophet_fcst, lstm_fcst], weights)
print(f"Error-weighted ensemble: {perf_weighted.round(2)}")
```

**Median Ensemble:** More robust to outliers than mean averaging:
```python
def median_ensemble(forecasts):
    return np.median(np.array(forecasts), axis=0)
```

**Stacking (Meta-Model):** Train a meta-model on validation predictions:
```python
from sklearn.linear_model import RidgeCV

def stacking_ensemble(base_models, X_train, y_train, X_val):
    """Train base models, then learn how to combine them with a meta-learner."""
    val_preds = np.column_stack([model.predict(X_val) for model in base_models])
    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta_model.fit(val_preds, y_val)
    return meta_model

# At inference:
# test_preds = np.column_stack([m.predict(X_test) for m in base_models])
# final = meta_model.predict(test_preds)
```

### 11a.3 Popular Time Series Ensemble Architectures

| Ensemble Type | How It Works | Pros | Cons | Competition Performance |
|--------------|-------------|------|------|:----------------------:|
| **Simple average** | Arithmetic mean of all models | Zero tuning, robust | All models weighted equally | M4: top 30% |
| **Error-weighted** | Inverse of validation RMSE | Rewards best models | Overfits to validation period | M4: top 15% |
| **Median** | Median across models | Outlier-robust | Loses information | M4: top 20% |
| **Stacking (Ridge)** | Ridge regression on model outputs | Automatically learns weights | Requires validation holdout | M4: top 10% |
| **Greedy ensemble** | Forward selection: add best model iteratively | Simple, effective | Sequential, may miss combos | M5: winning entry |
| **Bayesian ensemble** | Distribution over weights | Uncertainty-aware | Computationally intensive | Research |
| **Boosting of TS models** | Sequential fitting of residuals | Can improve weak models | May overfit | M5: top 10% |

### 11a.4 Greedy Ensemble (Winning M5 Approach)

The M5 competition was won by a greedy ensemble that sequentially selects models based on validation performance:

```python
def greedy_ensemble(models, X_val, y_val, max_models=5):
    """Forward selection ensemble — pick models that improve validation score."""
    from sklearn.metrics import mean_squared_error
    
    selected = []
    remaining = list(models)
    best_score = float('inf')
    
    for _ in range(min(max_models, len(models))):
        candidate_scores = []
        for model in remaining:
            current_preds = np.column_stack([m.predict(X_val) for m in selected + [model]])
            ensemble_pred = current_preds.mean(axis=1)
            score = mean_squared_error(y_val, ensemble_pred, squared=False)
            candidate_scores.append((score, model))
        
        candidate_scores.sort(key=lambda x: x[0])
        best_candidate_score, best_candidate = candidate_scores[0]
        
        if best_candidate_score < best_score:
            best_score = best_candidate_score
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break  # No improvement from adding more models
    
    return selected  # Use as: ensemble_pred = np.mean([m.predict(X) for m in selected], axis=0)
```

### 11a.5 When to Use Which Ensemble

| Scenario | Recommended Ensemble |
|----------|---------------------|
| **Quick baseline** | Simple average |
| **Clean validation set** | Error-weighted (inverse RMSE) |
| **Heterogeneous models** (ARIMA + Prophet + LSTM) | Median (robust to individual model failures) |
| **Large collection of similar models** | Greedy ensemble |
| **Competition / maximum accuracy** | Stacking (Ridge) + Greedy |
| **Uncertainty quantification needed** | Bayesian ensemble or quantile-averaging |
| **Computational budget limited** | Simple average or median |

### 11a.6 Practical Code: Ensemble Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import List, Callable

class TimeSeriesEnsemble:
    """Production-ready ensemble for time series forecasting."""
    
    def __init__(self, models: List[Callable], method: str = 'weighted'):
        self.models = models
        self.method = method
        self.weights = None
    
    def fit_weights(self, X_val, y_val):
        """Learn ensemble weights from validation data."""
        preds = np.column_stack([m.predict(X_val) for m in self.models])
        
        if self.method == 'equal':
            self.weights = np.ones(len(self.models)) / len(self.models)
        elif self.method == 'error_weighted':
            errors = np.array([mean_squared_error(y_val, p, squared=False) for p in preds.T])
            self.weights = (1/errors) / (1/errors).sum()
        elif self.method == 'ridge_stacking':
            from sklearn.linear_model import RidgeCV
            meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            meta.fit(preds, y_val)
            self.weights = meta.coef_
        
        return self
    
    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        if self.method in ['equal', 'error_weighted']:
            return np.dot(preds, self.weights)
        elif self.method == 'median':
            return np.median(preds, axis=1)
        elif self.method == 'ridge_stacking':
            return self.meta_model.predict(preds)
    
    def predict_interval(self, X, alpha=0.1):
        """Return prediction interval from ensemble spread."""
        preds = np.column_stack([m.predict(X) for m in self.models])
        point = self.predict(X)
        lower = np.percentile(preds, 100 * alpha/2, axis=1)
        upper = np.percentile(preds, 100 * (1 - alpha/2), axis=1)
        return point, lower, upper

# Usage
ensemble = TimeSeriesEnsemble(
    models=[arima_model, prophet_model, lstm_model, xgb_model],
    method='error_weighted'
)
ensemble.fit_weights(X_val, y_val)
forecast, lower, upper = ensemble.predict_interval(X_test, alpha=0.05)
```

**Key insight from M4/M5 competitions:** Ensemble methods consistently outperform individual models by 3-8% on forecasting accuracy (sMAPE/MASE). The combination of diverse model families (statistical + ML + DL) provides the greatest gains.

---

## 11b. Financial and Economic Time Series Applications

Financial time series present unique challenges: non-stationarity, volatility clustering, heavy tails, leverage effects, and regime changes. This section covers specialized models and techniques for financial and economic forecasting.

### 11b.1 Volatility Modeling (GARCH Family)

Financial returns exhibit **volatility clustering** — periods of high volatility tend to persist, as do periods of calm. The GARCH family of models captures this phenomenon.

| Model | Innovation | Equation | Best For |
|-------|-----------|----------|----------|
| **ARCH(q)** (Engle, 1982) | Autoregressive conditional heteroskedasticity | σ²_t = ω + Σα_i ε²_{t-i} | Basic volatility clustering detection |
| **GARCH(1,1)** (Bollerslev, 1986) | ARCH + lagged variance | σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} | General volatility forecasting |
| **EGARCH** (Nelson, 1991) | Log variance, asymmetric | ln(σ²_t) = ω + α(|z_{t-1}|-E|z_{t-1}|) + γ z_{t-1} + β ln(σ²_{t-1}) | Leverage effects (bad news → higher vol) |
| **GJR-GARCH** (Glosten et al., 1993) | Asymmetric term on negative shocks | σ²_t = ω + α ε²_{t-1} + γ I_{t-1} ε²_{t-1} + β σ²_{t-1} | Asymmetric volatility response |
| **APARCH** (Ding et al., 1993) | Power transformations | σ^δ_t = ω + α(|ε_{t-1}|-γ ε_{t-1})^δ + β σ^δ_{t-1} | Flexible power specification |
| **FIGARCH** (Baillie et al., 1996) | Fractional integration | (1-φL)(1-L)^d ε²_t = ω + (1-βL)ν_t | Long memory in volatility |

```python
import numpy as np, pandas as pd
from arch import arch_model

# Simulate returns with volatility clustering
np.random.seed(42)
n = 2000
omega, alpha, beta = 0.01, 0.1, 0.85
sigma2 = np.zeros(n)
returns = np.zeros(n)
sigma2[0] = omega / (1 - alpha - beta)
returns[0] = np.sqrt(sigma2[0]) * np.random.randn()

for t in range(1, n):
    sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

# Fit GARCH(1,1)
model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
result = model.fit(disp='off')
print(result.summary())

# Forecast volatility for next 10 days
forecasts = result.forecast(horizon=10)
print(f"Conditional volatility today: {result.conditional_volatility[-1]:.4f}")
print(f"Forecast volatility (10-day): {np.sqrt(forecasts.variance.values[-1, :]).round(4)}")
```

**Key insight:** The GARCH(1,1) with Student's t innovations is the workhorse model for financial volatility. It captures most of the volatility clustering effect with just three parameters (ω, α, β).

### 11b.2 Value at Risk (VaR) and Expected Shortfall

Risk forecasting is a critical application of time series in finance. Two key metrics:

| Metric | Definition | Interpretation | Regulation |
|--------|-----------|:-------------:|:----------:|
| **VaR_α** | Loss threshold exceeded with probability 1-α over horizon h | "We are 95% confident we will not lose more than VaR in one day" | Basel II/III (VaR 99%, 10-day) |
| **Expected Shortfall (CVaR)** | Average loss beyond VaR threshold | "If we exceed VaR, the average loss will be ES" | Basel III (ES 97.5%, replaces VaR) |
| **Drawdown** | Peak-to-trough decline | "Maximum cumulative loss from peak" | Portfolio monitoring |

**VaR Calculation Approaches:**

| Method | Description | Pros | Cons |
|--------|-------------|:----|:----|
| **Historical simulation** | Use empirical quantile of past returns | No distribution assumption | Assumes past ≈ future; needs 250+ days |
| **Parametric (variance-covariance)** | Assume normal distribution, VaR = μ + σ·z_α | Simple, fast | Fails on heavy-tailed returns |
| **Monte Carlo** | Simulate return paths from fitted model | Flexible, captures any distribution | Computationally intensive |
| **GARCH + filtered historical** | Fit GARCH, standardize returns, simulate from empirical residuals | Best coverage, captures vol clustering + fat tails | Complex implementation |

```python
def garch_var(returns, alpha=0.05, horizon=1, n_sim=10000):
    """Compute VaR and Expected Shortfall using GARCH + filtered historical simulation."""
    from arch import arch_model
    
    # Fit GARCH
    am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='studentst')
    res = am.fit(disp='off')
    
    # Standardized residuals (filtered historical distribution)
    std_resid = res.resid / res.conditional_volatility
    
    # Simulate future paths
    sigma_t = res.conditional_volatility.iloc[-1]
    sim_returns = []
    for _ in range(n_sim):
        shock = np.random.choice(std_resid, size=horizon)
        vol = sigma_t
        path_returns = []
        for t in range(horizon):
            r = vol * shock[t]
            path_returns.append(r)
            vol = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * r**2 + res.params['beta[1]'] * vol**2)
        sim_returns.append(np.sum(path_returns))
    
    sim_returns = np.array(sim_returns)
    VaR = np.percentile(sim_returns, alpha * 100)
    ES = sim_returns[sim_returns <= VaR].mean()
    return VaR / 100, ES / 100

returns = pd.Series(np.random.randn(500) * 0.02 - 0.0001)
VaR_95, ES_975 = garch_var(returns, alpha=0.05)
print(f"95% VaR (1-day): {VaR_95:.4f}  |  97.5% ES: {ES_975:.4f}")
```

### 11b.3 Regime-Switching Models

Economic and financial time series often exhibit distinct regimes (bull vs bear market, expansion vs recession, low vs high volatility). Regime-switching models capture this by allowing model parameters to change according to an unobserved state variable.

**Markov Switching Autoregressive (MS-AR) Model:**

```python
import statsmodels.api as sm
import numpy as np, pandas as pd

# Generate regime-switching data
np.random.seed(42)
n = 500
regime = np.zeros(n)
data = np.zeros(n)

# Regime 0: low mean, low volatility (bull market)
# Regime 1: high mean, high volatility (bear market rallies)
P = np.array([[0.95, 0.05], [0.10, 0.90]])  # transition matrix
means = [0.001, -0.002]
stds = [0.01, 0.03]

state = 0
for t in range(n):
    regime[t] = state
    data[t] = means[state] + stds[state] * np.random.randn()
    state = np.random.choice([0, 1], p=P[state])

# Fit Markov switching autoregressive model
try:
    ms_model = sm.tsa.MarkovRegression(data, k_regimes=2, trend='c', switching_variance=True)
    ms_result = ms_model.fit(disp=False)
    print(f"Regime 0 mean: {ms_result.params[0]:.4f}, Regime 1 mean: {ms_result.params[1]:.4f}")
    print(f"Transition P[0→0]: {ms_result.params[2]:.3f}, P[1→1]: {ms_result.params[3]:.3f}")
    predicted_regimes = ms_result.smoothed_marginal_probabilities[0].apply(lambda x: 0 if x > 0.5 else 1)
    regime_accuracy = np.mean(predicted_regimes == regime)
    print(f"Regime classification accuracy: {regime_accuracy:.2%}")
except Exception as e:
    print(f"Markov switching model error: {e}")
```

| Model Type | Application | Strengths | Limitations |
|------------|-------------|-----------|-------------|
| **MS-AR** (Hamilton, 1989) | Business cycle analysis | Captures recession/expansion asymmetry | Assumes fixed transition probabilities |
| **MS-GARCH** | Volatility regimes | Bull/bear volatility distinction | Estimation complexity, path dependence |
| **TVTP-MS** (time-varying transition) | Financial crises | Transition probabilities depend on leading indicators | More parameters, overfitting risk |
| **Threshold AR (TAR)** | Economic regimes | Observable threshold (e.g., unemployment rate) | Requires specifying threshold variable |
| **Smooth Transition AR (STAR)** | Gradual regime changes | Continuous transition between regimes | Difficult to interpret regimes |

### 11b.4 Practical Considerations for Financial TS

| Consideration | Why It Matters | Mitigation |
|:-------------|:---------------|:-----------|
| **Non-stationarity** | Financial returns are approximately stationary, but volatility is not | Use returns (not prices), fit GARCH for volatility |
| **Survivorship bias** | Backtesting on currently-listed assets overstates performance | Include delisted assets, use point-in-time data |
| **Look-ahead bias** | Using information not available at prediction time | Lag all features, use as-of datasets |
| **Microstructure noise** | Bid-ask bounce, stale prices at high frequencies | Use mid-quotes, aggregate to 5-min bars |
| **Regime changes** | Models trained in one regime fail in another | Regular retraining, regime detection triggers |
| **Tiny signal-to-noise** | Financial returns have very low predictability | Use multiple signals, shrinkage estimators |
| **Transaction costs** | Gross returns ≠ net returns | Include spread + commission in backtest |
| **Overfitting** | Countless strategies tested, few survive | Walk-forward CV, out-of-sample testing |
| **Multiple testing** | Thousands of assets × hundreds of strategies = many false positives | Bonferroni/FDR correction, purged CV |
| **Data snooping** | Same data used for hypothesis generation and testing | Holdout periods, synthetic data validation |

```python
# Walk-forward backtest framework for financial strategies
def walk_forward_backtest(prices, model_fn, train_window=252, test_window=21):
    """
    Walk-forward backtest with purged training windows.
    Prevents look-ahead by ensuring no test data leaks into training.
    """
    results = []
    start = 0
    while start + train_window + test_window <= len(prices):
        train = prices.iloc[start:start + train_window]
        test = prices.iloc[start + train_window:start + train_window + test_window]
        
        # Train model on purged window
        model = model_fn(train)
        
        # Generate out-of-sample predictions
        pred = model.predict(test)
        
        # Evaluate
        actual = test.pct_change().iloc[1:]
        if len(pred) == len(actual):
            results.append(np.corrcoef(pred, actual)[0, 1])
        
        start += test_window
    
    if results:
        avg_corr = np.mean(results)
        print(f"Walk-forward avg correlation: {avg_corr:.4f}")
        print(f"Stable periods: {sum(1 for r in results if r > 0)}/{len(results)} ({sum(1 for r in results if r > 0)/len(results):.0%})")
    return results

# Usage (simulated)
prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(1000) * 0.01)))
results = walk_forward_backtest(prices, lambda x: lambda y: np.full(len(y), 0.001))
```

**Key takeaway for financial TS:** The most important factor in financial forecasting is not model complexity — it is **rigorous out-of-sample evaluation** with proper purging, walk-forward CV, and realistic transaction costs. A simple moving average crossover with proper risk management often outperforms a sophisticated GARCH-LSTM that lacks rigorous backtesting.

| Application | Recommended Approach | Key Metric |
|:------------|:--------------------|:-----------|
| Equity return forecasting | Lasso/Ridge on macro factors | Information Coefficient (IC) |
| Volatility forecasting | GARCH(1,1) or GJR-GARCH with t-innovations | QLIKE loss, VaR backtest |
| Portfolio risk | RiskMetrics EWMA or DCC-GARCH | Tracking error, TEV |
| Macroeconomic nowcasting | Dynamic Factor Model (DFM) or MIDAS | RMSE, pseudo-R² |
| Credit risk | Logistic regression with duration features | AUC-ROC, Brier score |
| Commodity price forecasting | ARIMA + exogenous (weather, supply chain) | sMAPE, directional accuracy |
| Cryptocurrency (high-frequency) | Order book imbalance + XGBoost | Sharp ratio, max drawdown |

**See:** [10-Industry/02-AI-Economics.md] (§3 — Financial AI); [01-Foundations/02-Machine-Learning.md] (§6 — Model Evaluation)

---

## 12. Cross-References
| Reference | Description |
|-----------|-------------|
| [01-Foundations/02-Machine-Learning.md] | ML foundations |
| [01-Foundations/06-Reinforcement-Learning.md] | RL for inventory/forecasting control |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production forecasting pipelines |
| [10-Industry/01-AI-Industry-Applications.md] | Industry use cases (demand, energy) |
| [08-Reference/01-Glossary.md] | Key terms |
---
*Document version: 2.5 → 3.0 — June 2026 | Tier 2-3: Gap Fill | Expanded with code examples, evaluation metrics, stationarity testing, LSTM, feature engineering, transformer models, TS cross-validation, multi-step forecasting, causal/structural TS, deployment considerations, ensemble methods, and §11b Financial & Economic TS Applications (GARCH, VaR, regime-switching, walk-forward backtest)*
