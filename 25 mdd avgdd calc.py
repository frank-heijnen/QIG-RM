import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# The following are 20 tickers + benchmark. Since quant funds don't publish their data, I used quant-strategy ETFs which act as a sort of proxy for quant funds as they employ quant rules to form portfolios
funds = [
    "QCLN",  # First Trust NASDAQ Clean Edge Green Energy Index Fund
    "QARP",  # JPMorgan US Equity Factor ETF (Quant)
    "DTEC",  # Global X CleanTech ETF
    "QTUM",  # Defiance Quantum ETF
    "QAI",   # IQ Hedge Multi-Strategy Tracker ETF
    "QUS",   # U.S. Large Cap Equity Factor ETF
    "QQQM",  # Invesco NASDAQ 100 ETF
    "QVAL",  # Alpha Architect US Quantitative Value ETF
    "QUAL",  # iShares Edge MSCI USA Quality Factor ETF
    "QDIV",  # Global X S&P 500 Quality Dividend ETF
    "QDF",   # Global X S&P 500 Dividend ETF
    "QGEN",  # Global X Genomics & Biotechnology ETF
    "QYLD",  # Global X NASDAQ 100 Covered Call ETF
    "QRFT",  # Global X Robotics & AI ETF
    "QTEC",  # NASDAQ-100 Technology Sector Index ETF
    "QID",   # ProShares UltraShort QQQ
    "QLD",   # ProShares Ultra QQQ
    "QDEF",  # ALPS Quantitative Defensive Equity ETF
    "QED",   # Global X Artificial Intelligence ETF
    "JPUS"   # JPMorgan Diversified Return U.S. Equity ETF (replacement for QUSF)
]

benchmark = "SPY"  # S&P 500
all_tickers = funds + [benchmark]

# -----------------------------
# Download data from yahoo finance api
# -----------------------------
raw_data = yf.download(all_tickers, start="2015-01-01", end="2024-12-31", auto_adjust=True)

# Adjust for MultiIndex or normal DataFrame (done to fix an error i was getting)
if isinstance(raw_data.columns, pd.MultiIndex):
    data = raw_data['Close']
else:
    data = raw_data

# Drop tickers that failed to download (done to facilitate testing, right now all tickers download)
data = data.dropna(axis=1, how='all')

# Compute daily returns
returns = data.pct_change().dropna()

# Compute cumulative returns
cum_returns = (1 + returns).cumprod()

# -----------------------------
# Functions
def drawdown_series(series):
    cum_max = series.cummax()
    return (series - cum_max) / cum_max
# max drawdown is the highest fall in returns a stock has seen
def max_drawdown(series):
    return drawdown_series(series).min()

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
    fund_mdd = max_drawdown(fund_cum - benchmark_cum)
    fund_avg10 = avg_10_worst_drawdowns_rel_diff(fund_cum, benchmark_cum)

    # Relative returns (daily excess over SPY)
    rel_rets = fund_rets - spy_rets
    fund_var, fund_cvar = var_cvar(rel_rets)

    results.append({
        "Fund": fund,
        "Max Drawdown (rel to SPY)": fund_mdd,
        "Avg 10 Worst Drawdowns (rel to SPY %)": fund_avg10,
        "VaR 95% (rel)": fund_var,
        "CVaR 95% (rel)": fund_cvar
    })

df_results = pd.DataFrame(results)
print("\nRisk Metrics vs SPY:\n", df_results)

# -----------------------------
# Summary stats
# -----------------------------
avg_var = df_results["VaR 95% (rel)"].mean()
avg_cvar = df_results["CVaR 95% (rel)"].mean()
avg_avg10 = df_results["Avg 10 Worst Drawdowns (rel to SPY %)"].mean()

# Exclude QGEN outlier for avg as it was an individual stock and not an ETF
df_no_outlier = df_results[df_results["Fund"] != "QGEN"]
avg_avg10_no_outlier = df_no_outlier["Avg 10 Worst Drawdowns (rel to SPY %)"].mean()

print("\n--- Summary Stats ---")
print(f"Average VaR (95%, relative): {avg_var:.2%}")
print(f"Average CVaR (95%, relative): {avg_cvar:.2%}")
print(f"Average of Avg 10 Worst Drawdowns (all funds): {avg_avg10:.2f}%")
print(f"Average of Avg 10 Worst Drawdowns (excl QGEN): {avg_avg10_no_outlier:.2f}%")

# -----------------------------
# Plotting graph
plt.figure(figsize=(12, 7))
plt.scatter(df_results["Avg 10 Worst Drawdowns (rel to SPY %)"],
            df_results["Max Drawdown (rel to SPY)"], color="teal")

for i, row in df_results.iterrows():
    plt.text(row["Avg 10 Worst Drawdowns (rel to SPY %)"],
             row["Max Drawdown (rel to SPY)"], row["Fund"], fontsize=8)

plt.xlabel("Average of 10 Worst Drawdowns Relative to SPY (%)")
plt.ylabel("Maximum Drawdown Relative to SPY")
plt.title("Drawdown & Risk Analysis of Quant ETFs/Funds vs SPY")
plt.grid(True)
plt.tight_layout()
plt.show()


