import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests, io

# tried to format terminal columns properly
pd.set_option('display.max_columns', None)
# -----------------------------
# The following are 67 tickers + benchmark. 
# Since quant funds don't publish their data, I used quant-strategy ETFs which act as a sort of proxy for quant funds as they employ quant rules to form portfolios
#Most of them are obtained from the etfdb which categorizes them as "quant based etf"
funds = [
    "QCLN",  # First Trust NASDAQ Clean Edge Green Energy Index Fund
    "QARP",  # JPMorgan US Equity Factor ETF (Quant)
    "DTEC",  # Global X CleanTech ETF
    #"QTUM",  # Defiance Quantum ETF (outlier eliminated)
    "QAI",   # IQ Hedge Multi-Strategy Tracker ETF
    "QUS",   # U.S. Large Cap Equity Factor ETF
    "QQQM",  # Invesco NASDAQ 100 ETF
    "QVAL",  # Alpha Architect US Quantitative Value ETF
    "QUAL",  # iShares Edge MSCI USA Quality Factor ETF
    "QDIV",  # Global X S&P 500 Quality Dividend ETF
    "QDF",   # Global X S&P 500 Dividend ETF
    "GNOM",  # Global X Genomics & Biotechnology ETF
    "QYLD",  # Global X NASDAQ 100 Covered Call ETF
    "QRFT",  # Global X Robotics & AI ETF
    "QTEC",  # NASDAQ-100 Technology Sector Index ETF
    "QDEF",  # ALPS Quantitative Defensive Equity ETF
    "JPUS",   # JPMorgan Diversified Return U.S. Equity ETF
    "JEPQ",  # JPMorgan NASDAQ Equity Premium Income ETF
    "LRGF",  # iShares US Equity Factor ETF
    "INTF",  # iShares International Equity Factor ETF
    "SMLF",  # iShares U.S. Small-Cap Equity Factor ETF
    "JMEE",  # JPMorgan Small & Mid Cap Enhanced Equity ETF
    "IDMO",  # Invesco S&P International Developed Momentum ETF
    "DBMF",  # iMGP DBi Managed Futures Strategy ETF
    "ILOW",  # AB International Low Volatility Equity ETF
    "AVIG",  # Avantis Core Fixed Income ETF
    "RODM",  # Hartford Multifactor Developed Markets (ex-US) ETF
    "VFMO",  # Vanguard U.S. Momentum Factor ETF
    "QEFA",  # SPDR MSCI EAFE StrategicFactors ETF
    "SYLD",  # Cambria Shareholder Yield ETF
    "BUYW",  # Main BuyWrite ETF
    "SYFI",  # AB Short Duration High Yield ETF
    "OUSA",  # ALPS O'Shares U.S. Quality Dividend ETF
    "VFVA",  # Vanguard U.S. Value Factor ETF
    "QQH",   # HCM Defender 100 Index ETF
    "LRGC",  # AB US Large Cap Strategic Equities ETF
    "LGH",   # HCM Defender 500 Index ETF
    "FHEQ",  # Fidelity Hedged Equity ETF
    "ROUS",  # Hartford Multifactor US Equity ETF
    "ISCF",  # iShares International Small-Cap Equity Factor ETF
    "VFQY",  # Vanguard U.S. Quality Factor ETF
    "AFLG",  # First Trust Active Factor Large Cap ETF
    "VFMF",  # Vanguard U.S. Multifactor ETF
    "QVAL",  # Alpha Architect U.S. Quantitative Value ETF
    "FLSP",  # Franklin Systematic Style Premia ETF
    #"VFMV",  # Vanguard U.S. Minimum Volatility ETF (outlier eliminated)
    "TACK",  # Fairlead Tactical Sector ETF
    "GVLU",  # Gotham 1000 Value ETF
    "TUG",   # STF Tactical Growth ETF
    #"HYTR",  # Counterpoint High Yield Trend ETF (outlier eliminated)
    "TBG",   # TBG Dividend Focus ETF
    "CPLS",  # AB Core Plus Bond ETF
    "IVAL",  # Alpha Architect International Quantitative Value ETF
    "QWLD",  # SPDR MSCI World StrategicFactors ETF
    "SIHY",  # Harbor Scientific Alpha High-Yield ETF
    "AESR",  # Anfield U.S. Equity Sector Rotation ETF
    "HYBI",  # NEOS Enhanced Income Credit Select ETF
    "DWUS",  # AdvisorShares Dorsey Wright FSM US Core ETF
    "AIEQ",  # Amplify AI Powered Equity ETF
    "BMDL",  # VictoryShares WestEnd Economic Cycle Bond ETF
    "OWNS",  # CCM Affordable Housing MBS ETF
    "ACSI",  # American Customer Satisfaction ETF
    "TRFM",  # AAM Transformers ETF
    "LSAT",  # Leadershares Alphafactor Tactical Focused ETF
    "FFTY",  # Innovator IBD 50 ETF
    "ARB",    # AltShares Merger Arbitrage ETF
    "EPQ",   # JPMorgan NASDAQ Equity Premium Income ETF
    "MODL",  # VictoryShares WestEnd U.S. Sector ETF
    "AFMC",  # First Trust Active Factor Mid Cap ETF
    "DWAW",  # AdvisorShares Dorsey Wright FSM All Cap World ETF
    "AOTG",  # AOT Growth and Innovation ETF
    "REVS",  # Columbia Research Enhanced Value ETF
    "RAYD",  # Rayliant Quantitative Developed Market Equity ETF
    "RAYE",  # Rayliant Quantamental Emerging Market ex-China Equity ETF
    "FYEE",  # Fidelity Yield Enhanced Equity ETF
    "FDIV",  # MarketDesk Focused U.S. Dividend ETF
    "GTR",   # WisdomTree Target Range Fund
    "DRUP",  # GraniteShares Nasdaq Select Disruptors ETF
    "CAMX",  # Cambiar Aggressive Value ETF
    "PPEM",  # Putnam PanAgora ESG Emerging Markets Equity ETF
    "AFSM",  # First Trust Active Factor Small Cap ETF
    "ROAM",  # Hartford Multifactor Emerging Markets ETF
    "XRLV",  # Invesco S&P 500® ex-Rate Sensitive Low Volatility ETF
    "TXS",   # Texas Capital Texas Equity Index ETF
    "SLDR",  # Global X Short-Term Treasury Ladder ETF
    "COWS",  # Amplify Cash Flow Dividend Leaders ETF
    "RAYC",  # Rayliant Quuantamental China Equity ETF
    "SQEW",  # LeaderShares Equity Skew ETF
    "DWSH",  # AdvisorShares Dorsey Wright Short ETF
    "FBUF",  # Fidelity Dynamic Buffered Equity ETF
    "GLOW",  # VictoryShares WestEnd Global Equity ETF
    "LQAI",  # LG QRAFT AI-Powered U.S. Large Cap Core ETF
    "EMOP",  # AB Emerging Markets Opportunities ETF
    "SPCZ"   # RiverNorth Enhanced Pre-Merger SPAC ETF

]

benchmark = "SPY"  # S&P 500
all_tickers = funds + [benchmark]

# -----------------------------
# Download data from yahoo finance api
# -----------------------------
# Define target period
target_start = "2019-01-01"
target_end = "2025-09-30" 

# Download data with a buffer period to ensure we have enough historical data
buffer_start = "2015-01-01"  # Further back to ensure we capture the required history
raw_data = yf.download(all_tickers, start=buffer_start, end="2025-09-30", auto_adjust=True)

# Handle MultiIndex or normal DataFrame
if isinstance(raw_data.columns, pd.MultiIndex):
    data = raw_data['Close']
else:
    data = raw_data

print("Initial data shape:", data.shape)
print("Tickers downloaded:", data.columns.tolist())

# Filter for tickers that have data at our required starting point 
def has_sufficient_data(series, required_date="2019-01-31", min_required_date="2018-12-01"):
    """
    Check if a ticker has data available at the required date.
    If exact date is missing, check if data exists around that period.
    """
    required_date = pd.to_datetime(required_date)
    min_required_date = pd.to_datetime(min_required_date)
    
    # Check if there's any data in the required timeframe
    series_in_period = series.loc[min_required_date:required_date + pd.DateOffset(months=1)]
    return series_in_period.notna().sum() >= 1  # At least one data point in the period

# Apply the filter
valid_tickers = []
for ticker in data.columns:
    if has_sufficient_data(data[ticker]):
        valid_tickers.append(ticker)

print(f"Tickers with sufficient data: {valid_tickers}")

# Keep only valid tickers
filtered_data = data[valid_tickers]

# Ensure SPY is always included (force include if missing)
if benchmark not in filtered_data.columns:
    print(f"Warning: {benchmark} was filtered out. Forcing inclusion.")
    filtered_data[benchmark] = data[benchmark]
    # Drop if SPY has no data at all in our target period
    if not has_sufficient_data(filtered_data[benchmark]):
        print(f"Warning: {benchmark} has insufficient data but included anyway.")

print(f"Final tickers: {filtered_data.columns.tolist()}")

# Compute monthly returns for the target period
target_data = filtered_data.loc[target_start:target_end]
returns = target_data.resample('M').last().pct_change().dropna()
cum_returns = (1 + returns).cumprod()

print(f"Final returns data from {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
print(f"Tickers in final dataset: {returns.columns.tolist()}")
# -----------------------------
# Functions
def drawdown_series(series):
    cum_max = series.cummax()
    return (series - cum_max) / cum_max
# max drawdown is the highest fall in returns a stock has seen
def max_drawdown(series):
    return drawdown_series(series).min()*100
#defining relative drawdown where we simply subtract the S&P drawdown from it
def avg_10_worst_drawdowns_rel_diff(fund_series, benchmark_series):
    """Compute avg of 10 worst daily relative drawdowns (fund - benchmark)"""
    rel_diff = fund_series.pct_change() - benchmark_series.pct_change()
    return rel_diff.nsmallest(10).mean() * 100

def var_cvar(rel_returns, alpha=0.05):
    """Compute daily 1-alpha VaR and CVaR"""
    var = np.quantile(rel_returns, alpha)
    cvar = rel_returns[rel_returns <= var].mean()
    return var, cvar

# -----------------------------
# Benchmark stats for S&P
benchmark_cum = cum_returns[benchmark]
benchmark_dd_series = drawdown_series(benchmark_cum)
benchmark_mdd = benchmark_dd_series.min()
benchmark_avg10 = benchmark_dd_series.nsmallest(10).mean()

print(f"{benchmark} Max Drawdown: {benchmark_mdd:.2%}")
print(f"{benchmark} Avg of 10 Worst Drawdowns: {benchmark_avg10:.2%}")


#Finding risk free rate for sharpe ratio calculation using FRED data for 1 month Tbills (maybe unnecessary)

FRED_SERIES_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

def fetch_fred_series(series_id: str) -> pd.Series:
    url = FRED_SERIES_CSV.format(sid=series_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    date_col = [c for c in df.columns if "date" in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    s = pd.to_numeric(df[df.columns[0]], errors="coerce")
    return s.dropna()

#convert them to monthly returns
rf_annual = fetch_fred_series("DGS1MO") / 100  # 5% -> 0.05
rf_daily = (1 + rf_annual) ** (1/252) - 1
rf_monthly= (1+rf_daily).resample('M').prod()-1
rf_monthly=rf_monthly.reindex(data.resample('M').mean().index).ffill()
CE = rf_monthly

# -----------------------------
# Funds stats
results = []
for fund in funds:
    if fund not in cum_returns.columns:
        print(f"Skipping {fund}, data unavailable")
        continue

    fund_cum = cum_returns[fund]
    fund_rets = returns[fund]
    spy_rets = returns[benchmark]

    # Relative cumulative returns
    fund_mdd = max_drawdown(fund_cum / benchmark_cum)
    fund_avg10 = avg_10_worst_drawdowns_rel_diff(fund_cum, benchmark_cum)

    # Relative returns (daily excess over SPY)
    rel_rets = fund_rets - spy_rets
    fund_var, fund_cvar = var_cvar(rel_rets)

    #Sharpe Ratio for funds
   # --- Excess return over risk-free
    excess = fund_rets - CE  # relative to risk free rate
    mu = excess.mean()             # mean monthly excess return
    sigma = excess.std()        # monthly standard deviation
    excess_sp= fund_rets - spy_rets
    mu_sp = excess_sp.mean()
    sigma_sp = excess_sp.std()

    # --- Sharpe ratio (annualized)
    sharpe_ratio = (mu / sigma) * np.sqrt(12)
    #Sharpe ratio annualized relative to market (information ratio)
    sharpe_ratio_sp = (mu_sp/ sigma_sp) * np.sqrt(12)

    

    results.append({
        "Fund": fund,
        "Max Drawdown (rel to SPY)": fund_mdd,
        "Avg 10 Worst Drawdowns (rel to SPY %)": fund_avg10,
        "VaR 95% (rel)": fund_var,
        "CVaR 95% (rel)": fund_cvar,
        "Sharpe Ratio": sharpe_ratio,
        "Information Ratio": sharpe_ratio_sp
    })

df_results = pd.DataFrame(results)
print("\nRisk Metrics vs SPY:\n", df_results)

# -----------------------------
# Summary stats
# -----------------------------
avg_var = df_results["VaR 95% (rel)"].mean()
avg_cvar = df_results["CVaR 95% (rel)"].mean()
avg_avg10 = df_results["Avg 10 Worst Drawdowns (rel to SPY %)"].mean()
avg_mdd = df_results["Max Drawdown (rel to SPY)"].mean()
avg_SR = df_results["Sharpe Ratio"].mean()
avg_IR = df_results["Information Ratio"].mean()


print("\n--- Summary Stats ---")
print(f"Average VaR (95%, relative): {avg_var:.2%}")
print(f"Average CVaR (95%, relative): {avg_cvar:.2%}")
print(f"Average of Avg 10 Worst Drawdowns (all funds): {avg_avg10:.2f}%")
print(f"Average Max Drawdowns: {avg_mdd:.2f}%")
print(f"Average Sharpe Ratio: {avg_SR: .2f}")
print(f"Average Information Ratio: {avg_IR: .2f}")



# -----------------------------
# Plotting graph
plt.figure(figsize=(12, 7))
plt.scatter(df_results["Avg 10 Worst Drawdowns (rel to SPY %)"],
            df_results["Max Drawdown (rel to SPY)"], color="teal")

for i, row in df_results.iterrows():
    plt.text(row["Avg 10 Worst Drawdowns (rel to SPY %)"],
             row["Max Drawdown (rel to SPY)"], row["Fund"], fontsize=8)

plt.xlabel("Average of 10 Worst Drawdowns Relative to SPY (%)")
plt.ylabel("Maximum Drawdown Relative to SPY (%)")
plt.title("Drawdown & Risk Analysis of Quant ETFs/Funds vs SPY")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# Custom Portfolio Analysis
# -----------------------------
# QIG portfolio monthly returns (% to decimal)
my_rets = pd.Series(
    [-5.25, 3.17, 2.15, 1.21, 3.21, -0.14, 2.01, -3.63, 4.31, -8.18, -11.48, -0.66, 5.74, 0.005, -1.3, -0.55, 6.36,],
    index=pd.date_range("2024-04-30", periods=17, freq="M")
) / 100

# Align ETF and S&P returns to same dates
etf_monthly = returns.copy().resample("M").mean().reindex(my_rets.index, method="ffill")
spy_rets = etf_monthly[benchmark]
# Reindex them to match portfolio's dates (filling missing months)
etf_monthly = etf_monthly.reindex(my_rets.index).fillna(method="ffill")
spy_rets = spy_rets.reindex(my_rets.index).fillna(method="ffill")

# Calculate average ETF return
avg_etf_rets = etf_monthly.mean(axis=1)

# ===== Metrics =====
def sharpe_ratio(r):
    return (r.mean() / r.std()) * np.sqrt(12) if r.std() != 0 else np.nan

def info_ratio(rp, rb):
    diff = rp - rb
    return (diff.mean() / diff.std()) * np.sqrt(12) if diff.std() != 0 else np.nan

metrics = pd.DataFrame({
    "Sharpe Ratio": [
        sharpe_ratio(my_rets),
        sharpe_ratio(avg_etf_rets)
    ],
    "Info Ratio (vs SPY)": [
        info_ratio(my_rets, spy_rets),
        info_ratio(avg_etf_rets, spy_rets)
    ]
}, index=["My Portfolio", "Average ETF"])

print("\n--- Portfolio vs ETF Comparison ---")
print(metrics.round(3))

# ===== Chart =====
cum_df = pd.DataFrame({
    "My Portfolio": (1 + my_rets).cumprod() - 1,
    "Avg ETF": (1 + avg_etf_rets).cumprod() - 1,
    "SPY": (1 + spy_rets).cumprod() - 1
})

plt.figure(figsize=(10, 6))
plt.plot(cum_df.index, cum_df["My Portfolio"], label="My Portfolio", linewidth=2.2)
plt.plot(cum_df.index, cum_df["Avg ETF"], label="Average ETF", linestyle="--", linewidth=2)
plt.plot(cum_df.index, cum_df["SPY"], label="SPY", linestyle=":", linewidth=2)
plt.title("Cumulative Returns (May 2024 – Sep 2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
