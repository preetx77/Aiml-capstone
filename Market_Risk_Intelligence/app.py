"""Streamlit dashboard for the saved Market Risk Intelligence analysis."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT, OUT = Path(__file__).resolve().parent, Path(__file__).resolve().parent / "outputs"
COLORS = {"Low Volatility": "#22c55e", "Medium Volatility": "#f59e0b", "High Volatility": "#ef4444"}
st.set_page_config(page_title="Market Risk Intelligence", page_icon="📈", layout="wide")
st.markdown("""<style>.stApp{background:#07111f;color:#e5e7eb}.block-container{padding-top:1.5rem}.metric-card{padding:.6rem;border-left:3px solid #38bdf8;background:#0f1d31;border-radius:8px}</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    required = ["regime_results.csv", "regime_summary.csv", "transition_matrix.csv", "model_evaluation.csv", "cluster_profiles.csv"]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing: return None, missing
    results = pd.read_csv(OUT / "regime_results.csv", parse_dates=["Date"])
    tables = {name[:-4]: pd.read_csv(OUT / name, index_col=0 if name == "transition_matrix.csv" else None) for name in required[1:]}
    for optional in ["portfolio_returns.csv", "portfolio_risk.csv", "regime_risk_comparison.csv", "cluster_size_comparison.csv"]:
        if (OUT / optional).exists(): tables[optional[:-4]] = pd.read_csv(OUT / optional, parse_dates=["Date"] if optional == "portfolio_returns.csv" else None)
    return (results, tables), []

def window_transition(regimes: pd.Series) -> pd.DataFrame:
    """Transition probabilities within the selected chronological date window."""
    order = list(COLORS)
    current = pd.Series(regimes.iloc[:-1].to_numpy())
    following = pd.Series(regimes.iloc[1:].to_numpy())
    counts = pd.crosstab(current, following).reindex(index=order, columns=order, fill_value=0)
    return counts.div(counts.sum(axis=1).replace(0, pd.NA), axis=0)

def window_summary(frame: pd.DataFrame) -> pd.DataFrame:
    order = list(COLORS)
    summary = frame.groupby("Regime").agg(Observations=("Regime", "size"), Average_Volatility=("Volatility", "mean"), Average_VIX=("VIX", "mean")).reindex(order, fill_value=0).reset_index()
    summary["Percentage_of_Time"] = summary["Observations"] / len(frame) * 100
    return summary

def selected_portfolio_metrics(portfolio: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Rebase wealth and risk measures to the currently selected date window."""
    view = portfolio.copy().sort_values("Date")
    view["Wealth"] = (1 + view["Portfolio_Return"]).cumprod()
    view["Drawdown"] = view["Wealth"].div(view["Wealth"].cummax()).sub(1)
    r = view["Portfolio_Return"]
    var = r.quantile(.05)
    metrics = {"Annualized_Volatility": r.std() * (252 ** .5), "Sharpe_Ratio": r.mean() / r.std() * (252 ** .5) if r.std() else float("nan"), "VaR_95": var, "CVaR_95": r[r <= var].mean()}
    comparison = view.groupby("Regime")["Portfolio_Return"].agg(Daily_Volatility="std").reindex(list(COLORS)).reset_index()
    return view, metrics, comparison

loaded, missing = load_data()
if missing:
    st.error("Analysis outputs are missing: " + ", ".join(missing))
    st.code("python train_model.py", language="bash")
    st.stop()
results, tables = loaded
st.title("AI-Based Market Condition Detection and Risk Monitoring")
st.caption("Historical analysis only — values update for the selected saved-data range and are not real-time market quotes.")

st.sidebar.header("Analysis controls")
min_date, max_date = results.Date.min().date(), results.Date.max().date()
start = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="start_date")
end = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="end_date")
if start > end:
    st.sidebar.error("End date must be on or after the start date.")
    st.stop()
selected = results[results.Date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
if selected.empty:
    st.warning("No saved market observations exist in this range. Please choose another range.")
    st.stop()
st.sidebar.success(f"{len(selected):,} observations selected")
latest = selected.iloc[-1]
summary = window_summary(selected)

st.header("1. Market overview")
cols = st.columns(5)
for col, label, value in zip(cols, ["NIFTY 50", "India VIX", "Daily return", "20-day volatility", "Detected regime"], [f"{latest.NIFTY50:,.2f}", f"{latest.VIX:.2f}", f"{latest.Return:.2%}", f"{latest.Volatility:.2%}", latest.Regime]):
    col.metric(label, value)
st.markdown(f"### :{'green' if latest.Regime == 'Low Volatility' else 'orange' if latest.Regime == 'Medium Volatility' else 'red'}[ {latest.Regime.upper()} ]")
st.write(f"On **{latest.Date:%d %b %Y}**, NIFTY 50 was {latest.NIFTY50:,.2f}, India VIX was {latest.VIX:.2f}, and 20-day return volatility was {latest.Volatility:.2%}. This is a historical cluster assignment, not a forecast.")

st.header("2. Regime detection")
c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(px.scatter(selected, x="Date", y="NIFTY50", color="Regime", color_discrete_map=COLORS, title="Market-regime timeline for selected range", opacity=.8), use_container_width=True)
    st.caption("Each dot is one trading day. Its colour is the K-Means market condition assigned from return, volatility, momentum, VIX, and drawdown. Clusters describe historical similarity; they are not price forecasts.")
with c2:
    st.dataframe(summary.style.format({"Percentage_of_Time":"{:.1f}%", "Average_Volatility":"{:.2%}", "Average_VIX":"{:.2f}"}), hide_index=True, use_container_width=True)
    st.caption("This table summarises only the selected period. A larger average volatility or VIX generally indicates more uncertain market conditions.")

st.header("3. Historical analysis")
subset = selected
left, right = st.columns(2)
with left:
    st.plotly_chart(px.line(subset, x="Date", y="NIFTY50", title="NIFTY 50 historical price", markers=len(subset) < 30), use_container_width=True)
    st.caption("NIFTY 50 represents a broad basket of major Indian companies. Rising values show historical index gains; falling values show historical declines. It does not by itself measure risk.")
with right:
    st.plotly_chart(px.line(subset, x="Date", y="VIX", title="India VIX historical level", markers=len(subset) < 30), use_container_width=True)
    st.caption("India VIX is commonly interpreted as an uncertainty indicator. Higher values indicate that the market expected larger near-term price swings; it is not a direction forecast.")
transition = window_transition(selected["Regime"])
fig = px.imshow(transition, text_auto=".1%", color_continuous_scale="Blues", title="Chronological regime transition probabilities", labels={"x":"Next regime", "y":"Current regime", "color":"Probability"})
st.plotly_chart(fig, use_container_width=True)
st.caption("Each row shows where observed transitions from that current regime went next. For example, 15% from Low to Medium means 15% of observed Low-regime days were followed by a Medium-regime day. Historical probabilities are not guarantees.")
st.caption("The date controls above recalculate the timeline, regime distribution, transition matrix, and portfolio metrics for the selected period.")

st.header("4. Portfolio risk")
if "portfolio_returns" not in tables:
    st.warning("Portfolio files are unavailable. Supply cached `data/processed/stock_prices.csv` with the ten NSE symbols, or rerun training with a successful yfinance download.")
else:
    portfolio = tables["portfolio_returns"]
    portfolio = portfolio[portfolio.Date.between(pd.Timestamp(start), pd.Timestamp(end))]
    if len(portfolio) < 2:
        st.warning("Choose a wider date range to calculate portfolio risk metrics.")
    portfolio, risk, comparison = selected_portfolio_metrics(portfolio)
    metrics = st.columns(4)
    for col, label, value in zip(metrics, ["Annualised volatility", "Sharpe ratio", "VaR 95%", "CVaR 95%"], [f"{risk['Annualized_Volatility']:.2%}", f"{risk['Sharpe_Ratio']:.2f}", f"{risk['VaR_95']:.2%}", f"{risk['CVaR_95']:.2%}"]): col.metric(label, value)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.line(portfolio, x="Date", y="Wealth", title="Equal-weighted portfolio wealth"), use_container_width=True)
        st.caption("This begins at 1.0 on the selected start date and compounds the portfolio's daily returns. It illustrates historical growth or loss of one rupee invested, before costs and taxes.")
    with b:
        st.plotly_chart(px.line(portfolio, x="Date", y="Drawdown", title="Portfolio drawdown"), use_container_width=True)
        st.caption("Drawdown is the percentage fall from the portfolio's previous peak within this selected range. More negative values show a deeper historical decline.")
    st.plotly_chart(px.bar(comparison, x="Regime", y="Daily_Volatility", color="Regime", color_discrete_map=COLORS, title="Conditional daily volatility by regime"), use_container_width=True)
    st.caption("This compares the spread of portfolio daily returns on days assigned to each market regime. It describes risk conditional on historical labels, not the return of a trading strategy.")
    st.caption("Equal weights, 252 trading days/year, and zero risk-free rate are assumed; costs, taxes and slippage are excluded. Regime-wise values are conditional descriptions, not a continuous trading strategy.")

st.header("5. Model evaluation")
evaluation = tables["model_evaluation"]
e1, e2 = st.columns(2)
with e1:
    st.plotly_chart(px.line(evaluation, x="K", y="Inertia", markers=True, title="Elbow comparison"), use_container_width=True)
    st.caption("Inertia measures within-cluster variation. It always falls as more clusters are added, so the useful choice is where further reductions start becoming modest—the visual 'elbow'.")
with e2:
    st.plotly_chart(px.bar(evaluation, x="K", y="Train_Silhouette", title="Training silhouette comparison"), use_container_width=True)
    st.caption("A higher silhouette score means the training clusters are more compact and distinct. It measures grouping quality, not the ability to predict future prices.")
st.plotly_chart(px.scatter(results, x="PC1", y="PC2", color="Regime", color_discrete_map=COLORS, symbol="Dataset_Partition", title="PCA visualisation of training-fitted regimes"), use_container_width=True)
st.caption("PCA compresses the five scaled features into two display axes so separation can be inspected visually. PCA is only used for this chart; K-Means was trained on all five scaled features.")
st.dataframe(tables["cluster_profiles"], hide_index=True, use_container_width=True)
k3 = evaluation[evaluation.K.eq(3)].iloc[0]
st.info(f"K-Means groups similar scaled observations; it does not predict future prices. Scaling stops VIX or index-level features from dominating distances. PCA only projects the fitted feature space for display (PC1+PC2 explain {k3.PCA_Explained_Variance_PC1_PC2:.1%}). Silhouette measures cluster separation—not prediction accuracy. K=3 is retained for interpretable low/medium/high risk levels even if another K scores higher.")
