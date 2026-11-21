
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
OUT_DIR = "."


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def make_synthetic_data(n_days: int = 1095, seed: int = 42) -> pd.DataFrame:
    """Daily sales with trend + weekly/yearly seasonality + noise, plus
    deliberately injected dirt (missing values, duplicates, outliers)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    trend = 200 + 0.25 * t
    weekly = 25 * np.sin(2 * np.pi * dates.dayofweek / 7)
    yearly = 40 * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 8, n_days)
    sales = trend + weekly + yearly + noise

    df = pd.DataFrame({"Date": dates, "Sales": sales})
    df.loc[rng.choice(n_days, 15, replace=False), "Sales"] = np.nan          # missing
    df.loc[rng.choice(n_days, 6, replace=False), "Sales"] *= 3               # outliers
    df = pd.concat([df, df.sample(8, random_state=1)])                       # duplicates
    return df.sample(frac=1, random_state=3).reset_index(drop=True)          # shuffled


def load_data(csv, date_col, target_col):
    if csv:
        df = pd.read_csv(csv)
        print(f"Loaded {csv}: {df.shape}")
        return df, date_col, target_col
    print("No CSV given -> using synthetic sales data.")
    return make_synthetic_data(), "Date", "Sales"


# ----------------------------------------------------------------------------
# 2. CLEANING & PREPROCESSING
# ----------------------------------------------------------------------------
def clean(df, date_col, target_col, freq):
    df = df[[date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    n0 = len(df)
    df = df.drop_duplicates(subset=date_col, keep="first")
    print(f"Duplicates removed: {n0 - len(df)}")

    s = df.set_index(date_col)[target_col].sort_index()
    s = s.asfreq(freq)  # make the index regular (exposes hidden gaps as NaN)

    # Outliers: flag points far from a rolling median (robust MAD z-score)
    med = s.rolling(15, center=True, min_periods=3).median()
    resid = s - med
    mad = (resid - resid.median()).abs().median()
    robust_z = 0.6745 * (resid - resid.median()) / (mad if mad else 1)
    outliers = robust_z.abs() > 6
    print(f"Outliers treated: {int(outliers.sum())}")
    s[outliers] = np.nan

    print(f"Missing values interpolated: {int(s.isna().sum())}")
    s = s.interpolate(method="time").bfill().ffill()
    s.name = target_col
    return s


def add_features(s: pd.Series, lags=(1, 7, 14, 28), windows=(7, 28)) -> pd.DataFrame:
    """Lag, rolling and calendar features. Rolling stats are shifted by 1
    so they never peek at the value being predicted (no leakage)."""
    df = s.to_frame("y")
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std()
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["time_idx"] = np.arange(len(df))
    return df.dropna()


# ----------------------------------------------------------------------------
# 3. EXPLORATION
# ----------------------------------------------------------------------------
def explore(s: pd.Series, period: int):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s, lw=0.8)
    ax.plot(s.rolling(period).mean(), color="red", lw=2, label=f"{period}-step rolling mean")
    ax.set(title="Cleaned time series", ylabel=s.name)
    ax.legend()
    save(fig, "01_series.png")

    if len(s) >= 2 * period:
        dec = seasonal_decompose(s, model="additive", period=period)
        fig = dec.plot()
        fig.set_size_inches(12, 8)
        save(fig, "02_decomposition.png")
        trend = dec.trend.dropna()
        slope = (trend.iloc[-1] - trend.iloc[0]) / len(trend)
        print(f"Trend: {'upward' if slope > 0 else 'downward'} (~{slope:.3f} per step)")
        amp = dec.seasonal.max() - dec.seasonal.min()
        print(f"Seasonal amplitude (peak-to-trough): {amp:.2f}")


# ----------------------------------------------------------------------------
# 4/5. MODELS & EVALUATION
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred, name):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def time_split(df, test_frac=0.2):
    cut = int(len(df) * (1 - test_frac))  # chronological, NOT random
    return df.iloc[:cut], df.iloc[cut:]


def recursive_forecast(model, history: pd.Series, future_idx, feature_cols, lags, windows):
    """Multi-step forecast: predict one step, append it, rebuild features, repeat."""
    hist = history.copy()
    preds = []
    for ts in future_idx:
        hist.loc[ts] = np.nan
        row = {}
        for lag in lags:
            row[f"lag_{lag}"] = hist.iloc[-1 - lag]
        past = hist.iloc[:-1]
        for w in windows:
            row[f"roll_mean_{w}"] = past.iloc[-w:].mean()
            row[f"roll_std_{w}"] = past.iloc[-w:].std()
        row["month"], row["dayofweek"], row["dayofyear"] = ts.month, ts.dayofweek, ts.dayofyear
        row["time_idx"] = len(hist) - 1
        X = pd.DataFrame([row])[feature_cols]
        p = model.predict(X)[0]
        hist.iloc[-1] = p
        preds.append(p)
    return pd.Series(preds, index=future_idx)


# ----------------------------------------------------------------------------
# 6. VISUALIZATION HELPERS
# ----------------------------------------------------------------------------
def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--target-col", default="Sales")
    ap.add_argument("--freq", default="D", help="pandas freq: D, W, M ...")
    ap.add_argument("--period", type=int, default=None,
                    help="seasonal period (D->7, W->52, M->12 by default)")
    ap.add_argument("--horizon", type=int, default=60, help="future steps to forecast")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    period = args.period or {"D": 7, "W": 52, "M": 12, "MS": 12}.get(args.freq, 7)
    lags = tuple(l for l in (1, 7, 14, 28) if l < 3 * period) or (1,)
    windows = (min(7, period), min(28, 4 * period))

    # --- 1-2. load & clean ---
    raw, dcol, tcol = load_data(args.csv, args.date_col, args.target_col)
    s = clean(raw, dcol, tcol, args.freq)
    print(f"Clean series: {len(s)} points, {s.index.min().date()} -> {s.index.max().date()}")

    # --- 3. explore ---
    print("\n[Exploration]")
    explore(s, period)

    # --- features & time split ---
    feats = add_features(s, lags, windows)
    train, test = time_split(feats, args.test_frac)
    X_cols = [c for c in feats.columns if c != "y"]
    y_train, y_test = train["y"], test["y"]
    print(f"\nTrain: {len(train)} | Test: {len(test)} (split by time)")

    results, preds = [], {}

    # --- 4a. baselines ---
    preds["Naive (last value)"] = pd.Series(y_train.iloc[-1], index=test.index)
    seasonal_naive = s.shift(period).reindex(test.index)
    preds[f"Seasonal naive (t-{period})"] = seasonal_naive

    lin = LinearRegression().fit(train[X_cols], y_train)
    preds["Linear Regression"] = pd.Series(lin.predict(test[X_cols]), index=test.index)

    # --- 4b. stronger models ---
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(train[X_cols], y_train)
    # One-step-ahead predictions (uses true lags) - fair comparison to baselines
    preds["Random Forest (1-step)"] = pd.Series(rf.predict(test[X_cols]), index=test.index)

    print("Fitting SARIMA (may take a moment)...")
    sarima = SARIMAX(s.loc[:train.index[-1]], order=(1, 1, 1), trend="t",
                     seasonal_order=(1, 0, 1, period),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    preds["SARIMA"] = sarima.get_forecast(len(test)).predicted_mean.set_axis(test.index)

    # --- 5. evaluate ---
    for name, p in preds.items():
        ok = p.notna()
        results.append(metrics(y_test[ok], p[ok], name))
    res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n[Test-set metrics] (lower MAE/RMSE better, higher R2 better)")
    print(res.round(3).to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    best = res.iloc[0]["Model"]
    baseline_rmse = res.loc[res["Model"] == "Naive (last value)", "RMSE"].iloc[0]
    print(f"\nBest model: {best} "
          f"({100 * (1 - res.iloc[0]['RMSE'] / baseline_rmse):.1f}% lower RMSE than naive baseline)")

    # --- 6a. actual vs predicted ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_train.iloc[-4 * period:], color="gray", label="Train (tail)")
    ax.plot(y_test, color="black", lw=1.5, label="Actual")
    for name in ["Linear Regression", "Random Forest (1-step)", "SARIMA"]:
        ax.plot(preds[name], lw=1.2, alpha=0.85, label=name)
    ax.set(title="Actual vs predicted (test period)", ylabel=tcol)
    ax.legend()
    save(fig, "03_actual_vs_predicted.png")

    # --- 6b. metric comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    res.set_index("Model")[["MAE", "RMSE"]].plot.barh(ax=axes[0], title="Error (lower is better)")
    res.set_index("Model")["R2"].plot.barh(ax=axes[1], color="seagreen", title="R² (higher is better)")
    save(fig, "04_metrics.png")

    # --- 6c. feature importance ---
    imp = pd.Series(rf.feature_importances_, index=X_cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, title="Random Forest feature importance")
    save(fig, "05_feature_importance.png")

    # --- 6d. future forecast with confidence band (SARIMA refit on all data) ---
    final = SARIMAX(s, order=(1, 1, 1), trend="t", seasonal_order=(1, 0, 1, period),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = final.get_forecast(args.horizon)
    mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s.iloc[-8 * period:], label="History")
    ax.plot(mean, color="red", label=f"Forecast ({args.horizon} steps)")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=0.2, label="95% CI")
    ax.set(title="Future forecast (SARIMA)", ylabel=tcol)
    ax.legend()
    save(fig, "06_future_forecast.png")

    pd.DataFrame({"forecast": mean, "lower_95": ci.iloc[:, 0], "upper_95": ci.iloc[:, 1]}) \
        .to_csv(os.path.join(OUT_DIR, "future_forecast.csv"))
    print(f"\nDone. See the '{OUT_DIR}/' folder.")


if __name__ == "__main__":
    main()