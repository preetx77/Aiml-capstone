# AI-Based Market Condition Detection and Risk Monitoring System

An explainable B.Tech AIML capstone that groups historical Indian market observations into low-, medium-, and high-volatility regimes. It uses K-Means clustering on NIFTY 50 returns, rolling volatility, momentum, India VIX, and drawdown, then presents historical transition and equal-weighted portfolio risk analysis.

## Problem statement and objectives

Market conditions are difficult to communicate using a single index value. This project turns historical NIFTY 50 and India VIX observations into readable risk states, visualises their chronology, measures transitions between states, and describes the risk of a ten-stock NSE equal-weight portfolio. It does **not** predict future prices or guarantee future market states.

## Data and preprocessing

`data/raw/nifty50.csv` and `data/raw/india_vix.csv` are the supplied historical datasets. The pipeline also downloads adjusted historical closes for RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, LT, ITC, SBIN, BHARTIARTL, and AXISBANK when portfolio data is not cached. Dates are parsed, de-duplicated, sorted, and joined by their common trading days. Missing rolling-feature rows are removed explicitly; genuine financial outliers are not automatically deleted.

Features are daily NIFTY return, 20-day rolling standard deviation of returns, 20-day NIFTY momentum, India VIX, and drawdown from the running NIFTY peak.

## Methodology and evaluation

The earliest 80% of observations trains `StandardScaler` and K-Means; the latest 20% is only transformed and assigned with the fitted model. This avoids data leakage. K values 2–8 are compared with inertia, silhouette, and cluster size. Although two clusters can have a better silhouette, K=3 is retained so the result can be communicated as low, medium, and high volatility. Cluster IDs are mapped after fitting by training-period average rolling volatility.

Silhouette evaluates the compactness and separation of clusters, not predictive accuracy. PCA is used only to visualise the scaled feature space. Transition probabilities are calculated from the ordered historical regime sequence and are descriptive—not forecasts.

Portfolio metrics use the complete consecutive return series: annualised mean return, annualised volatility, zero-risk-free-rate Sharpe ratio, maximum drawdown, VaR 95%, and CVaR 95%. Regime-wise values are conditional daily descriptive statistics, not a tradable strategy. Assumptions: equal weights, 252 trading days/year, zero risk-free rate, and no transaction costs, taxes, or slippage.

## Project files

```
Market_Risk_Intelligence/
├── Market_Risk_Intelligence_Capstone.ipynb  # guided notebook
├── train_model.py                            # reproducible pipeline
├── app.py                                   # Streamlit dashboard
├── data/raw/                                # supplied NIFTY/VIX files
├── data/processed/                          # optional cached stock prices
├── models/                                  # saved scaler and K-Means model
└── outputs/                                 # saved tables consumed by app.py
```

## Installation and execution

From the `Market_Risk_Intelligence` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If stock downloading is unavailable, build and demonstrate the regime-analysis portion without fabricating portfolio results:

```powershell
python train_model.py --skip-stock-download
```

To use the notebook, start Jupyter in this directory and open `Market_Risk_Intelligence_Capstone.ipynb`. It invokes the same pipeline, so notebook and dashboard use identical saved outputs.

## Dashboard sections

1. Market overview with latest saved historical observation and plain-language condition.
2. Regime timeline, distribution, volatility, and VIX summaries.
3. Date-filtered NIFTY/VIX history, transition heatmap, and average duration.
4. Portfolio wealth, drawdown, and risk comparison when stock data is available.
5. Elbow/silhouette charts, PCA visualisation, profiles, and methodology explanation.

## Limitations and future improvements

The result depends on historical features and a simple K-Means distance model; a cluster label is not a causal explanation or investment recommendation. Future work could add robust data-quality monitoring, controlled benchmark comparisons, sector allocations, walk-forward retraining, and macroeconomic explanatory variables.
