"""Config-driven backtest engine.

Usage
-----
    python -m backtesting.backtesting_framework backtesting/configs/momentum.yml

or programmatically::

    from backtesting.backtesting_framework import run_from_config
    result = run_from_config('backtesting/configs/momentum.yml')

Flow
----
    1. Load YAML config.
    2. Load price history via DataStore.
    3. Dynamically import the strategy class; instantiate with its own
       parameters only.
    4. Call strategy.generate_signals(prices) to get target weights.
    5. Normalise weights to respect max_leverage.
    6. Apply rebalance cadence (daily / weekly / monthly / ... ): outside
       rebalance days, weights stay constant so no turnover is charged.
    7. Compute daily gross returns from lagged weights and asset returns,
       subtract fees + slippage proportional to turnover.
    8. Produce summary stats and write everything to the configured
       output directory together with a copy of the config.

Design notes & deliberate deviations from backtesting_report.tex (v1)
--------------------------------------------------------------------
  - `rebalance_frequency` lives under `backtest:` rather than
    `strategy.parameters:`. It is an engine concern; strategies that
    implement `generate_signals` do not need it.
  - `data.universe` is a list of tickers (e.g. AAPL, MSFT, ...) rather
    than a named universe like "sp500". Universe registries are an
    obvious v2 addition.
  - `execution.trade_at_open` and `risk.vol_target` are accepted in the
    config but not enforced yet; they are documented as planned knobs.
  - Data source is yfinance only. `source: local_parquet` will raise.

See changes.txt for the full rationale.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from backtesting.data_store import DataStore
from backtesting.strategy import Strategy


# ---------------------------------------------------------------------------
# Config + strategy loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _import_strategy(class_name: str) -> type:
    """
    Dynamically import a strategy class from backtesting.strategies.

    By convention the module name is the class name lower-cased, so
    class `MomentumStrategy` lives in `backtesting/strategies/momentumstrategy.py`.
    That's ugly; we also accept a shorter module name (e.g. `momentum` for
    `MomentumStrategy`) by stripping a trailing 'Strategy' suffix.
    """
    candidates = [class_name.lower()]
    if class_name.lower().endswith("strategy"):
        candidates.append(class_name.lower()[: -len("strategy")])

    last_err: Exception | None = None
    for module_suffix in candidates:
        try:
            module = importlib.import_module(f"backtesting.strategies.{module_suffix}")
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            last_err = e
    raise ImportError(
        f"Could not import strategy {class_name!r} from backtesting.strategies "
        f"(tried modules: {candidates}). Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    summary: dict


class BacktestEngine:
    def __init__(self, config: dict):
        self.cfg = config
        self.store = DataStore(config.get("data", {}).get("source", "yfinance"))

    def run(self, strategy: Strategy) -> BacktestResult:
        data_cfg = self.cfg["data"]
        bt_cfg = self.cfg["backtest"]
        exec_cfg = self.cfg.get("execution", {}) or {}
        risk_cfg = self.cfg.get("risk", {}) or {}

        prices = self.store.load_universe(
            tickers=data_cfg["universe"],
            start=bt_cfg["start"],
            end=bt_cfg["end"],
            freq=data_cfg.get("freq", "1D"),
        )
        prices = prices.dropna(how="all").ffill().dropna()
        if prices.empty:
            raise ValueError("DataStore returned no overlapping price history.")

        raw_signals = strategy.generate_signals(prices)
        raw_signals = raw_signals.reindex(index=prices.index,
                                          columns=prices.columns).fillna(0.0)

        normalised = _normalise_weights(raw_signals, risk_cfg)
        target_weights = _apply_rebalance(
            normalised,
            bt_cfg.get("rebalance_frequency", "1D"),
        )

        asset_returns = prices.pct_change().fillna(0.0)
        held_weights = target_weights.shift(1).fillna(0.0)
        gross_returns = (held_weights * asset_returns).sum(axis=1)

        turnover = target_weights.diff().abs().sum(axis=1).fillna(0.0)
        cost_rate = (float(exec_cfg.get("fees_bps", 0))
                     + float(exec_cfg.get("slippage_bps", 0))) / 10_000.0
        costs = turnover * cost_rate

        net_returns = gross_returns - costs
        initial_capital = float(bt_cfg.get("initial_capital", 1.0))
        equity = initial_capital * (1.0 + net_returns).cumprod()

        summary = _compute_summary(equity, net_returns, turnover, target_weights)

        return BacktestResult(
            equity=equity,
            returns=net_returns,
            weights=target_weights,
            turnover=turnover,
            costs=costs,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Weight / rebalance helpers
# ---------------------------------------------------------------------------

def _normalise_weights(w: pd.DataFrame, risk_cfg: dict) -> pd.DataFrame:
    """Scale each row so gross exposure equals at most max_leverage."""
    max_lev = float(risk_cfg.get("max_leverage", 1.0))
    gross = w.abs().sum(axis=1)
    scale = max_lev / gross.replace(0, np.nan)
    scale = scale.clip(upper=1.0 if max_lev <= 1.0 else None)
    return w.mul(scale.fillna(0.0), axis=0)


def _rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> pd.Series:
    """Boolean series, True on the first trading day of each period."""
    s = pd.Series(dates, index=dates)
    if freq == "1D":
        return pd.Series(True, index=dates)
    if freq == "1W":
        key = s.apply(lambda d: (d.isocalendar().year, d.isocalendar().week))
    elif freq == "1M":
        key = s.apply(lambda d: (d.year, d.month))
    elif freq == "1Q":
        key = s.apply(lambda d: (d.year, d.quarter))
    elif freq == "1Y":
        key = s.apply(lambda d: d.year)
    else:
        raise ValueError(f"Unsupported rebalance_frequency {freq!r}")
    return ~key.duplicated(keep="first")


def _apply_rebalance(w: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Hold weights constant between rebalance dates."""
    mask = _rebalance_mask(w.index, freq)
    held = w.where(mask).ffill()
    return held.fillna(0.0)


# ---------------------------------------------------------------------------
# Summary stats + output
# ---------------------------------------------------------------------------

def _compute_summary(
    equity: pd.Series,
    returns: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
) -> dict:
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    ann_ret = float((1.0 + returns.mean()) ** 252 - 1.0)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": total_ret,
        "annualised_return": ann_ret,
        "annualised_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "avg_turnover_per_day": float(turnover.mean()),
        "n_rebalances": int((turnover > 0).sum()),
        "n_assets_average_long": float((weights > 0).sum(axis=1).mean()),
        "n_assets_average_short": float((weights < 0).sum(axis=1).mean()),
    }


def _write_results(out_dir: Path, cfg_path: Path, result: BacktestResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out_dir / "config.yml")
    result.equity.rename("equity").to_csv(out_dir / "equity.csv")
    result.returns.rename("returns").to_csv(out_dir / "returns.csv")
    result.weights.to_csv(out_dir / "weights.csv")
    result.turnover.rename("turnover").to_csv(out_dir / "turnover.csv")
    result.costs.rename("costs").to_csv(out_dir / "costs.csv")
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "summary": result.summary,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_from_config(config_path: str | Path) -> BacktestResult:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    StratClass = _import_strategy(cfg["strategy"]["class"])
    strategy = StratClass(**(cfg["strategy"].get("parameters") or {}))
    engine = BacktestEngine(cfg)
    result = engine.run(strategy)
    out_path = Path(cfg.get("output", {}).get("path", "results/unnamed"))
    _write_results(out_path, config_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m backtesting.backtesting_framework <config.yml>")
        return 1
    result = run_from_config(argv[0])
    print(json.dumps(result.summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
