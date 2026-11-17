"""Professional Streamlit interface for saved Market Risk Intelligence outputs."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT, OUT = Path(__file__).resolve().parent, Path(__file__).resolve().parent / "outputs"
COLORS = {"Low Volatility": "#20e3d2", "Medium Volatility": "#ffbf3f", "High Volatility": "#ff5c7a"}
ORDER, PLOT_CONFIG = list(COLORS), {"displayModeBar": False, "responsive": True}
st.set_page_config(page_title="MR Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
px.defaults.template = "plotly_dark"
st.markdown("""<style>
#MainMenu,footer,header{visibility:hidden}.stApp{background:#05080d;color:#e8f1f5}.block-container{max-width:1450px;padding:1.25rem 2.4rem 3rem}[data-testid="stSidebar"]{background:linear-gradient(180deg,#09111a,#05080d);border-right:1px solid #18303b}[data-testid="stSidebar"]>div:first-child{padding-top:1.3rem}h1,h2,h3{letter-spacing:-.035em!important;color:#f4fbfd!important}h1{font-size:2.35rem!important;margin-bottom:.1rem!important}h2{font-size:1.35rem!important;margin:1.7rem 0 .7rem!important}.eyebrow{color:#20e3d2;font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.muted{color:#91a5ad;font-size:.91rem}[data-testid="stMetric"]{background:#0a1119;border:1px solid #182a34;border-radius:13px;padding:1rem 1.1rem}[data-testid="stMetricLabel"]{color:#8fa2aa!important;font-size:.76rem!important;text-transform:uppercase;letter-spacing:.08em}[data-testid="stMetricValue"]{color:#f5fbfd!important;font-size:1.48rem!important}div[data-testid="stPlotlyChart"]{background:#080e15;border:1px solid #172934;border-radius:13px;overflow:hidden;min-height:420px}[data-testid="stDataFrame"]{border:1px solid #172934;border-radius:12px;overflow:hidden}.explain{color:#8fa4ac;font-size:.86rem;line-height:1.55;padding:.45rem .2rem .9rem}.condition{border:1px solid #1a3d48;border-left:4px solid #20e3d2;background:#071218;border-radius:10px;padding:.85rem 1rem;margin:.7rem 0 1rem}.condition b{color:#20e3d2;letter-spacing:.08em}.stButton>button,[data-testid="stDownloadButton"] button{border-radius:9px;border:1px solid #20e3d2!important;background:transparent!important;color:#20e3d2!important}.stButton>button:hover,[data-testid="stDownloadButton"] button:hover{background:#12323a!important}div[role="radiogroup"]{gap:.35rem;background:#081018;padding:.3rem;border:1px solid #172934;border-radius:11px}div[role="radiogroup"] label{padding:.35rem .65rem;border-radius:7px}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    required = ["regime_results.csv", "model_evaluation.csv", "cluster_profiles.csv"]
    missing = [n for n in required if not (OUT / n).exists()]
    if missing: return None, missing
    data = {"results": pd.read_csv(OUT / "regime_results.csv", parse_dates=["Date"]), "evaluation": pd.read_csv(OUT / "model_evaluation.csv"), "profiles": pd.read_csv(OUT / "cluster_profiles.csv")}
    for name in ["portfolio_returns.csv", "portfolio_risk.csv", "regime_risk_comparison.csv"]:
        if (OUT / name).exists(): data[name[:-4]] = pd.read_csv(OUT / name, parse_dates=["Date"] if name == "portfolio_returns.csv" else None)
    return data, []

def caption(text): st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)
def chart(fig):
    # Explicit height prevents later Streamlit components from overlapping Plotly
    # canvases after custom CSS is applied.
    fig.update_layout(height=400, autosize=True, paper_bgcolor="#080e15", plot_bgcolor="#080e15", font=dict(color="#c5d5da", family="Arial"), margin=dict(l=20,r=20,t=48,b=42), title_font=dict(size=16,color="#e9faff"), hoverlabel=dict(bgcolor="#10202a",font_color="#f3feff"))
    fig.update_xaxes(gridcolor="#13232d", zerolinecolor="#13232d"); fig.update_yaxes(gridcolor="#13232d", zerolinecolor="#13232d")
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
def summary(frame):
    out = frame.groupby("Regime").agg(Observations=("Regime","size"),Average_Volatility=("Volatility","mean"),Average_VIX=("VIX","mean")).reindex(ORDER,fill_value=0).reset_index(); out["Percentage_of_Time"] = out.Observations.div(len(frame)).mul(100); return out
def transitions(regimes):
    counts = pd.crosstab(pd.Series(regimes.iloc[:-1].to_numpy()),pd.Series(regimes.iloc[1:].to_numpy())).reindex(index=ORDER,columns=ORDER,fill_value=0); return counts.div(counts.sum(axis=1).replace(0,pd.NA),axis=0)
def portfolio_view(frame):
    out=frame.sort_values("Date").copy(); out["Wealth"]=(1+out.Portfolio_Return).cumprod(); out["Drawdown"]=out.Wealth.div(out.Wealth.cummax()).sub(1); r=out.Portfolio_Return; var=r.quantile(.05)
    risk={"vol":r.std()*252**.5,"sharpe":r.mean()/r.std()*252**.5 if r.std() else float("nan"),"var":var,"cvar":r[r<=var].mean()}; compare=out.groupby("Regime").Portfolio_Return.std().reindex(ORDER).rename("Daily_Volatility").reset_index(); return out,risk,compare

data, missing = load_data()
if missing:
    st.error("Required analysis outputs are missing: " + ", ".join(missing)); st.code("python train_model.py",language="powershell"); st.stop()
results = data["results"]
with st.sidebar:
    st.markdown("<div class='eyebrow'>MR INTELLIGENCE</div>",unsafe_allow_html=True); st.markdown("### Historical risk monitor"); st.markdown("<p class='muted'>Explore saved market observations. Date changes update all historical and portfolio views.</p>",unsafe_allow_html=True); st.divider()
    min_date,max_date=results.Date.min().date(),results.Date.max().date(); start=st.date_input("Start date",min_date,min_value=min_date,max_value=max_date); end=st.date_input("End date",max_date,min_value=min_date,max_value=max_date)
    if start>end: st.error("End date must be on or after start date."); st.stop()
    selected=results[results.Date.between(pd.Timestamp(start),pd.Timestamp(end))].copy(); st.success(f"{len(selected):,} observations selected"); st.download_button("Download selected data",selected.to_csv(index=False).encode("utf-8"),"market_regimes_selected.csv","text/csv"); st.markdown("<p class='muted'>Historical analysis only. No chart is an investment recommendation or a real-time signal.</p>",unsafe_allow_html=True)
if selected.empty: st.warning("There are no saved observations in that date range."); st.stop()
latest,selected_summary=selected.iloc[-1],summary(selected)
st.markdown("<div class='eyebrow'>AI-BASED MARKET CONDITION DETECTION</div>",unsafe_allow_html=True); st.title("Market Risk Intelligence"); st.markdown("<p class='muted'>Explainable historical regimes, transitions, and equal-weight portfolio risk.</p>",unsafe_allow_html=True)
page=st.radio("Navigate",["Overview","Regimes","Portfolio risk","Model evaluation"],horizontal=True,label_visibility="collapsed")

if page=="Overview":
    st.header("Market snapshot"); cols=st.columns(5); values=[f"{latest.NIFTY50:,.2f}",f"{latest.VIX:.2f}",f"{latest.Return:.2%}",f"{latest.Volatility:.2%}",latest.Regime]
    for col,label,value in zip(cols,["NIFTY 50","India VIX","Daily return","20-day volatility","Detected regime"],values): col.metric(label,value)
    accent=COLORS[latest.Regime]; st.markdown(f"<div class='condition' style='border-left-color:{accent}'><b style='color:{accent}'>{latest.Regime.upper()}</b><br>Latest selected observation: {latest.Date:%d %b %Y}. The label reflects the historical feature pattern on this date, not a prediction of tomorrow's market.</div>",unsafe_allow_html=True)
    left,right=st.columns(2)
    with left: chart(px.line(selected,x="Date",y="NIFTY50",title="NIFTY 50 historical price",markers=len(selected)<30)); caption("NIFTY 50 represents a broad basket of major Indian companies. The line shows historical index movement in the selected window; it is not a direct measure of risk.")
    with right: chart(px.line(selected,x="Date",y="VIX",title="India VIX historical level",markers=len(selected)<30,color_discrete_sequence=["#20e3d2"])); caption("India VIX is a market-uncertainty indicator. Higher levels generally indicate expectations of larger near-term price swings, not whether prices will rise or fall.")

if page=="Regimes":
    st.header("Market-regime detection"); left,right=st.columns([1.8,1])
    with left: chart(px.scatter(selected,x="Date",y="NIFTY50",color="Regime",color_discrete_map=COLORS,title="Regime timeline",opacity=.82)); caption("Each dot is one trading day. K-Means assigns its colour by comparing daily return, rolling volatility, momentum, VIX, and drawdown with other historical observations.")
    with right: st.markdown("#### Selected-period distribution"); st.dataframe(selected_summary.style.format({"Percentage_of_Time":"{:.1f}%","Average_Volatility":"{:.2%}","Average_VIX":"{:.2f}"}),hide_index=True,use_container_width=True); caption("The table updates with the chosen dates. High VIX and high rolling volatility typically occur more often in the higher-risk regime.")
    chart(px.imshow(transitions(selected.Regime),text_auto=".1%",color_continuous_scale=[[0,"#09141c"],[.5,"#117e88"],[1,"#20e3d2"]],title="Chronological transition probabilities",labels={"x":"Next regime","y":"Current regime","color":"Probability"})); caption("A row describes what followed a regime on the next observed trading day. For example, a Low-to-Medium value of 7% means 7% of Low-regime observations were followed by Medium. These are historical frequencies, not guarantees.")

if page=="Portfolio risk":
    st.header("Equal-weighted portfolio risk")
    if "portfolio_returns" not in data: st.warning("Portfolio output is unavailable. Run `python train_model.py` with valid stock-price data to enable this page.")
    else:
        portfolio=data["portfolio_returns"]; portfolio=portfolio[portfolio.Date.between(pd.Timestamp(start),pd.Timestamp(end))]
        if len(portfolio)<2: st.warning("Choose at least two trading observations for portfolio analysis.")
        else:
            portfolio,risk,compare=portfolio_view(portfolio); cols=st.columns(4)
            for col,label,value in zip(cols,["Annualised volatility","Sharpe ratio","VaR 95%","CVaR 95%"],[f"{risk['vol']:.2%}",f"{risk['sharpe']:.2f}",f"{risk['var']:.2%}",f"{risk['cvar']:.2%}"]): col.metric(label,value)
            left,right=st.columns(2)
            with left: chart(px.line(portfolio,x="Date",y="Wealth",title="Portfolio wealth",color_discrete_sequence=["#20e3d2"])); caption("Starts at 1.0 on the selected start date and compounds daily returns of the ten-stock equal-weight portfolio. It excludes transaction costs, taxes, and slippage.")
            with right: chart(px.area(portfolio,x="Date",y="Drawdown",title="Portfolio drawdown",color_discrete_sequence=["#ff5c7a"])); caption("Drawdown is the percentage decline from the previous portfolio peak inside the selected period. More negative values indicate a deeper historical fall.")
            chart(px.bar(compare,x="Regime",y="Daily_Volatility",color="Regime",color_discrete_map=COLORS,title="Conditional daily volatility by regime")); caption("This compares the variability of portfolio returns on days classified into each regime. It is a conditional description of history, not a back-tested switching strategy.")
            st.info("Assumptions: equal weighting, 252 trading days/year, zero risk-free rate for Sharpe ratio, and no transaction costs, taxes, or slippage.")

if page=="Model evaluation":
    st.header("Model evaluation"); st.markdown("<p class='muted'>These charts describe the saved original training experiment, so they do not change with the dashboard date filter.</p>",unsafe_allow_html=True); left,right=st.columns(2)
    with left: chart(px.line(data["evaluation"],x="K",y="Inertia",markers=True,title="Elbow comparison",color_discrete_sequence=["#20e3d2"])); caption("Inertia is within-cluster variation. It always decreases as more clusters are added; an 'elbow' suggests where additional clusters provide less extra structure.")
    with right: chart(px.bar(data["evaluation"],x="K",y="Train_Silhouette",title="Training silhouette comparison",color_discrete_sequence=["#20e3d2"])); caption("Higher silhouette values indicate more compact, separate clusters. It evaluates grouping quality, not future market-price accuracy.")
    chart(px.scatter(results,x="PC1",y="PC2",color="Regime",symbol="Dataset_Partition",color_discrete_map=COLORS,title="PCA visualisation of training-fitted regimes",opacity=.72)); caption("PCA compresses the five scaled model features into two visual axes. It is only for visual inspection; K-Means itself uses all five scaled features.")
    st.markdown("#### Cluster feature profile"); st.dataframe(data["profiles"],hide_index=True,use_container_width=True); k3=data["evaluation"].loc[data["evaluation"].K.eq(3)].iloc[0]; st.info(f"K=3 is retained for readable Low/Medium/High risk communication, even though another K may have a higher silhouette score. The first two PCA components explain {k3.PCA_Explained_Variance_PC1_PC2:.1%} of the scaled-feature variation.")
