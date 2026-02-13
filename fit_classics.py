"""
priors_setup.py

Builds returns-calibrated (Option A) rolling priors for:
- Merton jump-diffusion: (sigma, lam, mu_j, sig_j) via mixture likelihood (SciPy)
- Heston: (kappa, theta, sigma, rho, v0) via proxy calibration from realized variance

Main entry:
    storm_df_out = add_heston_merton_priors(
        storm_df,
        price_dir="options_data",
        price_col="Close",
        window=252,
        rv_window=20,
        merton_max_n=8,
        merton_step=5,
        heston_step=1,
    )

Notes:
- Uses only stock prices (per ticker CSVs in price_dir), no option data.
- Dates not present in the priors index are matched with "asof" (previous trading day).
- Requires SciPy for Merton. If SciPy is missing, Merton columns are filled with NaN.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm import tqdm

TRADING_DAYS = 252


# -----------------------------
# Data access / returns
# -----------------------------
def read_price_series(
    ticker: str,
    price_dir: str,
    price_col: str = "Close",
) -> pd.Series:
    """
    Reads a CSV located at: {price_dir}/{ticker}.csv
    Must contain a price column and a date index (parseable by pandas).
    """
    df = pd.read_csv(f"{price_dir}/{ticker}.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in {price_dir}/{ticker}.csv")
    s = df[price_col].astype(float).copy()
    s.name = "close"
    return s


def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.dropna()
    r = np.log(close).diff().dropna()
    r.name = "logret"
    return r


# -----------------------------
# Heston proxy priors
# -----------------------------
@dataclass(frozen=True)
class HestonPriors:
    kappa: float
    theta: float
    sigma: float  # vol of vol
    rho: float
    v0: float


def fit_heston_proxy_window(
    r_win: np.ndarray,
    dt: float,
    rv_window: int = 20,
) -> HestonPriors:
    """
    Returns-only proxy calibration:
      vhat_t = annualized rolling variance of returns within window
      Fit AR(1) on vhat to approximate mean reversion
    """
    r_win = np.asarray(r_win, dtype=float)
    if r_win.size < rv_window + 10:
        raise ValueError("Window too short for Heston proxy calibration.")

    # proxy annual variance
    rv = pd.Series(r_win).rolling(rv_window).var().dropna().to_numpy()
    vhat = TRADING_DAYS * rv  # annualized variance proxy

    x = vhat[:-1]
    y = vhat[1:]

    # OLS: y = a + b x + e
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    # stabilize b to (0,1) for mean-reversion
    b = min(max(b, 1e-6), 0.999999)

    kappa = -math.log(b) / dt
    theta = a / (1.0 - b)
    theta = max(theta, 1e-12)

    e = y - (a + b * x)

    # Approx: Var(e_t) ~ sigma^2 * v_t * dt
    denom = np.maximum(x, 1e-12) * dt
    sig2 = float(np.mean((e * e) / denom))
    sigma = math.sqrt(max(sig2, 1e-12))

    # rho proxy: corr between returns and variance changes
    dv = np.diff(vhat)
    if dv.size >= 3:
        r_for_corr = r_win[-dv.size:]
        rho = float(np.corrcoef(r_for_corr, dv)[0, 1])
        rho = float(np.clip(rho, -0.999, 0.999))
    else:
        rho = 0.0

    v0 = float(vhat[-1])

    return HestonPriors(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)


def rolling_heston_proxy_priors(
    logret: pd.Series,
    window: int = 252,
    rv_window: int = 20,
    step: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date with columns:
      h_kappa, h_theta, h_sigma, h_rho, h_v0
    Computed every `step` days and forward-filled to daily frequency.
    """
    dt = 1.0 / TRADING_DAYS
    r = logret.values
    dates = logret.index

    rows = []
    idx = []

    window_range = range(window, len(logret), step)
    if show_progress:
        window_range = tqdm(window_range, desc="Heston windows", leave=False)
    for i in window_range:
        pri = fit_heston_proxy_window(r[i - window: i], dt=dt, rv_window=rv_window)
        rows.append([pri.kappa, pri.theta, pri.sigma, pri.rho, pri.v0])
        idx.append(dates[i])

    df = pd.DataFrame(
        rows,
        index=pd.Index(idx, name="date"),
        columns=["h_kappa", "h_theta", "h_sigma", "h_rho", "h_v0"],
    ).sort_index()

    # forward-fill to all trading days in logret index (for asof joins)
    df = df.reindex(logret.index, method="ffill")
    return df


# -----------------------------
# Merton priors (SciPy likelihood)
# -----------------------------
@dataclass(frozen=True)
class MertonPriors:
    sigma: float  # diffusion vol (annualized)
    lam: float    # jump intensity per year
    mu_j: float   # mean log jump
    sig_j: float  # std log jump


def _scipy_minimize():
    try:
        from scipy.optimize import minimize  # type: ignore
        return minimize
    except Exception:
        return None


def fit_merton_window(
    r: np.ndarray,
    dt: float,
    max_n: int = 8,
    init: Optional[Tuple[float, float, float, float]] = None,
) -> MertonPriors:
    """
    Mixture likelihood (truncated at max_n jumps/day).
    Drift is not estimated (set to 0) since pricing overwrites drift with r-q (Option A).
    """
    minimize = _scipy_minimize()
    if minimize is None:
        raise RuntimeError("SciPy not available; cannot fit Merton via likelihood.")

    r = np.asarray(r, dtype=float)
    if r.size < 60:
        raise ValueError("Need at least ~60 returns for Merton window fit.")

    s0 = float(np.std(r) / math.sqrt(dt))
    if init is None:
        init = (max(1e-4, s0), 0.5, -0.02, 0.10)  # sigma, lam, mu_j, sig_j

    def nll(x: np.ndarray) -> float:
        sigma, lam, mu_j, sig_j = float(x[0]), float(x[1]), float(x[2]), float(x[3])

        if sigma <= 1e-8 or lam < 0.0 or sig_j <= 1e-8 or lam > 80.0:
            return 1e12

        lamdt = lam * dt

        # Vectorized log Poisson weights: logw[n] = -lamdt + sum_{k=1..n}(log(lamdt)-log(k))
        n_vals = np.arange(max_n + 1, dtype=float)
        log_lamdt = np.log(max(lamdt, 1e-30))
        logw = -lamdt + n_vals * log_lamdt - np.concatenate(
            [[0], np.cumsum(np.log(np.maximum(np.arange(1, max_n + 1, dtype=float), 1)))]
        )

        kappa_J = math.exp(mu_j + 0.5 * sig_j * sig_j) - 1.0
        mean_base = (-0.5 * sigma * sigma - lam * kappa_J) * dt
        var_base = (sigma * sigma) * dt

        mean_n = mean_base + n_vals * mu_j
        var_n = var_base + n_vals * (sig_j * sig_j)
        r_col = r[:, np.newaxis]
        log_density = -0.5 * (
            np.log(2 * np.pi * var_n) + (r_col - mean_n) ** 2 / var_n
        )
        comps = logw + log_density
        ll = float(np.sum(logsumexp(comps, axis=1)))
        return -ll

    x0 = np.array(init, dtype=float)
    bounds = [(1e-6, 5.0), (0.0, 80.0), (-2.0, 2.0), (1e-6, 3.0)]
    res = minimize(nll, x0=x0, bounds=bounds, method="L-BFGS-B")
    sigma, lam, mu_j, sig_j = map(float, res.x)
    return MertonPriors(sigma=sigma, lam=lam, mu_j=mu_j, sig_j=sig_j)


def rolling_merton_priors(
    logret: pd.Series,
    window: int = 252,
    max_n: int = 8,
    step: int = 5,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date with columns:
      m_sigma, m_lam, m_mu_j, m_sig_j
    Computed every `step` days and forward-filled to daily frequency.
    """
    dt = 1.0 / TRADING_DAYS
    r = logret.values
    dates = logret.index

    rows = []
    idx = []

    # warm start
    init: Optional[Tuple[float, float, float, float]] = None

    window_range = range(window, len(logret), step)
    if show_progress:
        window_range = tqdm(window_range, desc="Merton windows", leave=False)
    for i in window_range:
        pri = fit_merton_window(r[i - window: i], dt=dt, max_n=max_n, init=init)
        init = (pri.sigma, pri.lam, pri.mu_j, pri.sig_j)
        rows.append([pri.sigma, pri.lam, pri.mu_j, pri.sig_j])
        idx.append(dates[i])

    df = pd.DataFrame(
        rows,
        index=pd.Index(idx, name="date"),
        columns=["m_sigma", "m_lam", "m_mu_j", "m_sig_j"],
    ).sort_index()

    df = df.reindex(logret.index, method="ffill")
    return df


# -----------------------------
# Worker for parallel per-ticker processing
# -----------------------------
def _process_one_ticker(
    ticker: str,
    price_dir: str,
    price_col: str,
    window: int,
    rv_window: int,
    merton_max_n: int,
    merton_step: int,
    heston_step: int,
    scipy_ok: bool,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """Process a single ticker: read prices, compute rolling Heston and Merton priors. Used with ProcessPoolExecutor."""
    close = read_price_series(ticker, price_dir=price_dir, price_col=price_col)
    logret = compute_log_returns(close)
    hdf = rolling_heston_proxy_priors(
        logret,
        window=window,
        rv_window=rv_window,
        step=heston_step,
        show_progress=False,
    )
    if scipy_ok:
        try:
            mdf = rolling_merton_priors(
                logret,
                window=window,
                max_n=merton_max_n,
                step=merton_step,
                show_progress=False,
            )
        except Exception:
            mdf = pd.DataFrame(
                index=logret.index,
                data=np.nan,
                columns=["m_sigma", "m_lam", "m_mu_j", "m_sig_j"],
            )
    else:
        mdf = pd.DataFrame(
            index=logret.index,
            data=np.nan,
            columns=["m_sigma", "m_lam", "m_mu_j", "m_sig_j"],
        )
    return (ticker, hdf, mdf)


# -----------------------------
# Join priors into storm_data
# -----------------------------
def _asof_join_priors(
    storm_df: pd.DataFrame,
    priors_df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Efficient asof join per ticker (previous date match).
    """
    out_parts = []
    storm_df = storm_df.copy()
    storm_df[date_col] = pd.to_datetime(storm_df[date_col])

    for tkr, g in storm_df.groupby(ticker_col, sort=False):
        g = g.sort_values(date_col).copy()

        # priors_df is indexed by date for this ticker; convert to column for merge_asof
        p = priors_df.copy()
        p = p.reset_index().rename(columns={"index": date_col})
        p[date_col] = pd.to_datetime(p[date_col])
        p = p.sort_values(date_col)

        merged = pd.merge_asof(
            g,
            p,
            on=date_col,
            direction="backward",
            allow_exact_matches=True,
        )
        out_parts.append(merged)

    return pd.concat(out_parts, axis=0).sort_index()


def add_heston_merton_priors(
    storm_df: pd.DataFrame,
    price_dir: str,
    price_col: str = "Close",
    window: int = 252,
    rv_window: int = 20,
    merton_max_n: int = 8,
    merton_step: int = 5,
    heston_step: int = 1,
    date_col: str = "date",
    ticker_col: str = "ticker",
    verbose: bool = True,
    show_progress: bool = True,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    Reads per-ticker stock prices, calibrates rolling priors, and merges them into storm_df.

    Adds columns:
      Heston: h_kappa, h_theta, h_sigma, h_rho, h_v0
      Merton: m_sigma, m_lam, m_mu_j, m_sig_j  (NaN if SciPy missing)

    Returns a new DataFrame.
    """
    storm_df = storm_df.copy()
    storm_df[date_col] = pd.to_datetime(storm_df[date_col])

    tickers = storm_df[ticker_col].dropna().unique().tolist()

    # store per-ticker priors
    heston_by_ticker: Dict[str, pd.DataFrame] = {}
    merton_by_ticker: Dict[str, pd.DataFrame] = {}

    minimize = _scipy_minimize()
    scipy_ok = minimize is not None

    ticker_iter: Any = tickers
    if show_progress and n_workers <= 1:
        ticker_iter = tqdm(tickers, desc="Tickers")

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _process_one_ticker,
                    tkr,
                    price_dir,
                    price_col,
                    window,
                    rv_window,
                    merton_max_n,
                    merton_step,
                    heston_step,
                    scipy_ok,
                ): tkr
                for tkr in tickers
            }
            completed = as_completed(futures)
            if show_progress:
                completed = tqdm(completed, total=len(tickers), desc="Tickers")
            for future in completed:
                tkr, hdf, mdf = future.result()
                heston_by_ticker[tkr] = hdf
                merton_by_ticker[tkr] = mdf
        if verbose and not scipy_ok:
            print("[priors_setup] SciPy not found; Merton priors will be NaN.")
    else:
        for tkr in ticker_iter:
            if verbose:
                print(f"[priors_setup] ticker={tkr}")

            close = read_price_series(tkr, price_dir=price_dir, price_col=price_col)
            logret = compute_log_returns(close)

            hdf = rolling_heston_proxy_priors(
                logret,
                window=window,
                rv_window=rv_window,
                step=heston_step,
                show_progress=show_progress,
            )
            heston_by_ticker[tkr] = hdf

            if scipy_ok:
                try:
                    mdf = rolling_merton_priors(
                        logret,
                        window=window,
                        max_n=merton_max_n,
                        step=merton_step,
                        show_progress=show_progress,
                    )
                except Exception as e:
                    if verbose:
                        print(f"[priors_setup] Merton failed for {tkr}: {e}")
                    mdf = pd.DataFrame(
                        index=logret.index,
                        data=np.nan,
                        columns=["m_sigma", "m_lam", "m_mu_j", "m_sig_j"],
                    )
            else:
                if verbose:
                    print("[priors_setup] SciPy not found; Merton priors will be NaN.")
                mdf = pd.DataFrame(
                    index=logret.index,
                    data=np.nan,
                    columns=["m_sigma", "m_lam", "m_mu_j", "m_sig_j"],
                )
            merton_by_ticker[tkr] = mdf

    # merge per ticker
    out_parts = []
    groupby_iter = list(storm_df.groupby(ticker_col, sort=False))
    if show_progress:
        groupby_iter = tqdm(groupby_iter, desc="Merging priors")
    for tkr, g in groupby_iter:
        g = g.sort_values(date_col).copy()

        hdf = heston_by_ticker[tkr]
        mdf = merton_by_ticker[tkr]

        # asof join: need priors date column (index name is "date" in rolling priors)
        hdf_right = hdf.reset_index().rename(columns={hdf.index.name or "index": date_col})
        mdf_right = mdf.reset_index().rename(columns={mdf.index.name or "index": date_col})
        h_join = pd.merge_asof(
            g.sort_values(date_col),
            hdf_right,
            on=date_col,
            direction="backward",
            allow_exact_matches=True,
        )
        hm_join = pd.merge_asof(
            h_join.sort_values(date_col),
            mdf_right,
            on=date_col,
            direction="backward",
            allow_exact_matches=True,
        )
        out_parts.append(hm_join)

    out = pd.concat(out_parts, axis=0).sort_index()
    return out


if __name__ == "__main__":
    df = pd.read_csv("storm_data.csv")
    df = add_heston_merton_priors(df, price_dir="options_data", price_col="Close")
    df.to_csv("storm_data_with_priors.csv", index=False)
