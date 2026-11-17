"""Reproducible training pipeline for market-regime and portfolio-risk analysis.

Run from this project directory: python train_model.py
The script uses the local NIFTY/VIX CSV files and tries to download the ten NSE
stocks only when a local prices file is not available. It never uses test rows
to fit the scaler or K-Means model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RAW, PROCESSED, OUTPUTS, MODELS = (ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "outputs", ROOT / "models")
FEATURES = ["Return", "Volatility", "Momentum", "VIX", "Drawdown"]
REGIME_ORDER = ["Low Volatility", "Medium Volatility", "High Volatility"]
TICKERS = {"RELIANCE": "RELIANCE.NS", "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS", "INFY": "INFY.NS", "TCS": "TCS.NS", "LT": "LT.NS", "ITC": "ITC.NS", "SBIN": "SBIN.NS", "BHARTIARTL": "BHARTIARTL.NS", "AXISBANK": "AXISBANK.NS"}


def ensure_directories() -> None:
    for directory in (RAW, PROCESSED, OUTPUTS, MODELS): directory.mkdir(parents=True, exist_ok=True)


def read_market_file(path: Path, value_name: str, dayfirst: bool) -> pd.DataFrame:
    """Read either comma- or tab-delimited market files with Date and Price/Close."""
    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    frame.columns = frame.columns.str.replace('"', '', regex=False).str.strip()
    date_col = next((c for c in frame if c.lower() in {"date", "time"}), None)
    price_col = next((c for c in frame if c.lower() in {"price", "close"}), None)
    if not date_col or not price_col:
        raise ValueError(f"{path.name} needs Date and Price/Close columns; found {frame.columns.tolist()}")
    frame = frame[[date_col, price_col]].rename(columns={date_col: "Date", price_col: value_name})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", dayfirst=dayfirst, format="mixed")
    frame[value_name] = pd.to_numeric(frame[value_name].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return frame.dropna().drop_duplicates("Date").sort_values("Date").set_index("Date")


def download_stock_prices() -> pd.DataFrame:
    """Download adjusted close prices. Raises a clear error when network data is unavailable."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required to download stock prices. Install requirements.txt.") from exc
    prices = pd.DataFrame()
    for name, ticker in TICKERS.items():
        downloaded = yf.download(ticker, start="2016-10-26", end="2026-10-27", auto_adjust=True, progress=False)
        if not downloaded.empty:
            prices[name] = downloaded["Close"].squeeze()
    if prices.empty or len(prices.columns) != len(TICKERS):
        missing = sorted(set(TICKERS) - set(prices.columns))
        raise RuntimeError(f"Stock download incomplete. Missing: {missing}. No portfolio outputs were fabricated.")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    return prices.sort_index()


def load_stock_prices(download: bool) -> pd.DataFrame | None:
    local = PROCESSED / "stock_prices.csv"
    if local.exists():
        frame = pd.read_csv(local, parse_dates=["Date"]).set_index("Date").sort_index()
        if set(TICKERS).issubset(frame.columns): return frame[list(TICKERS)]
    if not download: return None
    prices = download_stock_prices()
    prices.to_csv(local)
    return prices


def make_features(nifty: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    data = nifty.join(vix, how="inner").sort_index().replace([np.inf, -np.inf], np.nan)
    data["Return"] = data["NIFTY50"].pct_change()
    data["Volatility"] = data["Return"].rolling(20).std()
    data["Momentum"] = data["NIFTY50"].pct_change(20)
    data["Drawdown"] = data["NIFTY50"] / data["NIFTY50"].cummax() - 1
    return data.dropna(subset=FEATURES).copy()


def safe_silhouette(x: np.ndarray, labels: Iterable[int]) -> float:
    labels = np.asarray(labels)
    return float(silhouette_score(x, labels)) if len(x) > 2 and len(np.unique(labels)) > 1 else np.nan


def transition_tables(regimes: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Use positional arrays. Pandas otherwise aligns the shifted date indexes and
    # incorrectly compares each observation with itself, producing a 100% diagonal.
    previous = pd.Series(regimes.iloc[:-1].to_numpy(), name="Current_Regime")
    following = pd.Series(regimes.iloc[1:].to_numpy(), name="Next_Regime")
    counts = pd.crosstab(previous, following).reindex(index=REGIME_ORDER, columns=REGIME_ORDER, fill_value=0)
    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    runs = regimes.ne(regimes.shift()).cumsum()
    run_frame = pd.DataFrame({"Run": runs.to_numpy(), "Regime": regimes.to_numpy()})
    duration = run_frame.groupby("Run").agg(Regime=("Regime", "first"), Days=("Regime", "size")).groupby("Regime").Days.mean().reindex(REGIME_ORDER)
    return counts, probabilities, duration.rename("Average_Consecutive_Days").reset_index()


def portfolio_outputs(results: pd.DataFrame, prices: pd.DataFrame | None) -> bool:
    if prices is None:
        return False
    aligned = results.join(prices, how="inner")
    stock_returns = aligned[list(TICKERS)].pct_change()
    portfolio = stock_returns.mean(axis=1, skipna=False).rename("Portfolio_Return")
    portfolio = pd.DataFrame({"Portfolio_Return": portfolio, "Regime": aligned["Regime"]}).dropna()
    wealth = (1 + portfolio.Portfolio_Return).cumprod()
    portfolio["Wealth"] = wealth
    portfolio["Drawdown"] = wealth.div(wealth.cummax()).sub(1)
    r = portfolio.Portfolio_Return
    var = r.quantile(.05)
    overall = pd.DataFrame([{"Annualized_Mean_Return": r.mean()*252, "Annualized_Volatility": r.std()*np.sqrt(252), "Sharpe_Ratio": r.mean()/r.std()*np.sqrt(252) if r.std() else np.nan, "Maximum_Drawdown": portfolio.Drawdown.min(), "VaR_95": var, "CVaR_95": r[r <= var].mean(), "Observations": len(r)}])
    rows = []
    for regime, group in portfolio.groupby("Regime"):
        values, conditional_var = group.Portfolio_Return, group.Portfolio_Return.quantile(.05)
        rows.append({"Regime": regime, "Observations": len(values), "Average_Daily_Return": values.mean(), "Daily_Volatility": values.std(), "VaR_95": conditional_var, "CVaR_95": values[values <= conditional_var].mean()})
    pd.DataFrame(rows).set_index("Regime").reindex(REGIME_ORDER).reset_index().to_csv(OUTPUTS / "regime_risk_comparison.csv", index=False)
    portfolio.reset_index(names="Date").to_csv(OUTPUTS / "portfolio_returns.csv", index=False)
    overall.to_csv(OUTPUTS / "portfolio_risk.csv", index=False)
    return True


def train(download_stocks: bool = True) -> None:
    ensure_directories()
    nifty = read_market_file(RAW / "nifty50.csv", "NIFTY50", dayfirst=True)
    vix = read_market_file(RAW / "india_vix.csv", "VIX", dayfirst=False)
    data = make_features(nifty, vix)
    split = int(len(data) * .8)
    train_data, test_data = data.iloc[:split].copy(), data.iloc[split:].copy()
    scaler = StandardScaler().fit(train_data[FEATURES])
    train_x, test_x = scaler.transform(train_data[FEATURES]), scaler.transform(test_data[FEATURES])
    # Retain K=3 for understandable three-level risk reporting; selection metrics remain transparent.
    evaluations, cluster_sizes = [], []
    for k in range(2, 9):
        candidate = KMeans(n_clusters=k, random_state=42, n_init=10).fit(train_x)
        labels = candidate.labels_
        evaluations.append({"K": k, "Inertia": candidate.inertia_, "Train_Silhouette": safe_silhouette(train_x, labels)})
        cluster_sizes.extend({"K": k, "Cluster": cluster, "Observations": count} for cluster, count in pd.Series(labels).value_counts().sort_index().items())
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(train_x)
    train_data["Cluster"], test_data["Cluster"] = model.labels_, model.predict(test_x)
    profile = train_data.groupby("Cluster")[FEATURES].mean().sort_values("Volatility")
    label_map = dict(zip(profile.index, REGIME_ORDER))
    for partition, frame in (("Train", train_data), ("Test", test_data)):
        frame["Regime"] = frame.Cluster.map(label_map)
        frame["Dataset_Partition"] = partition
    results = pd.concat([train_data, test_data]).sort_index()
    results.index.name = "Date"
    profiles = results.groupby("Cluster")[FEATURES].mean()
    profiles["Regime"] = profiles.index.map(label_map)
    profiles["Training_Observations"] = train_data.Cluster.value_counts()
    profiles.reset_index().to_csv(OUTPUTS / "cluster_profiles.csv", index=False)
    summary = results.groupby("Regime").agg(Average_Daily_Return=("Return", "mean"), Average_Volatility=("Volatility", "mean"), Average_VIX=("VIX", "mean"), Average_Drawdown=("Drawdown", "mean"), Observations=("Regime", "size")).reindex(REGIME_ORDER).reset_index()
    summary["Percentage_of_Time"] = summary.Observations / len(results) * 100
    counts, probabilities, durations = transition_tables(results.Regime)
    summary = summary.merge(durations, on="Regime", how="left")
    summary.to_csv(OUTPUTS / "regime_summary.csv", index=False)
    counts.to_csv(OUTPUTS / "transition_counts.csv")
    probabilities.to_csv(OUTPUTS / "transition_matrix.csv")
    pca = PCA(n_components=2).fit(train_x)
    all_x = np.vstack([train_x, test_x])
    components = pca.transform(all_x)
    results["PC1"], results["PC2"] = components[:, 0], components[:, 1]
    results.reset_index().to_csv(OUTPUTS / "regime_results.csv", index=False)
    evaluation = pd.DataFrame(evaluations)
    evaluation["Test_Silhouette_K3"] = np.nan
    evaluation.loc[evaluation.K.eq(3), "Test_Silhouette_K3"] = safe_silhouette(test_x, test_data.Cluster)
    evaluation["PCA_Explained_Variance_PC1_PC2"] = np.nan
    evaluation.loc[evaluation.K.eq(3), "PCA_Explained_Variance_PC1_PC2"] = pca.explained_variance_ratio_.sum()
    evaluation.to_csv(OUTPUTS / "model_evaluation.csv", index=False)
    pd.DataFrame(cluster_sizes).to_csv(OUTPUTS / "cluster_size_comparison.csv", index=False)
    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(model, MODELS / "kmeans_model.pkl")
    prices = load_stock_prices(download_stocks)
    portfolio_created = portfolio_outputs(results, prices)
    print(f"Trained on {len(train_data)} rows; evaluated on {len(test_data)} later rows.")
    print(f"Outputs written to the '{OUTPUTS.name}' directory. Portfolio outputs created: {portfolio_created}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-stock-download", action="store_true", help="Build regime outputs only when no cached stock prices exist.")
    args = parser.parse_args()
    train(download_stocks=not args.skip_stock_download)
