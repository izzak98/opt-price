"""
Optimized inference script for qStorm option pricing model.

This script performs batched inference on options data using the qStorm model,
with significant performance optimizations including vectorization, batching,
and efficient data handling.
"""

import argparse
import logging
import os
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from scipy.stats import norm
from tqdm import tqdm

from q_storm.qStorm import QStorm
from utils.data_utils import get_test_dataset
from varpi.QLSTM import LSTM_Model
from varpi.tain_varpi import VarPi

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

accelerator = Accelerator()


class InferenceConfig:
    """Configuration class for inference parameters."""

    def __init__(
        self,
        q_storm_model_path: str = "models/q_storm.pth",
        varphi_model_path: str = "models/varphi.pth",
        varpi_model_path: str = "models/varpi.pth",
        options_data_path: str = "options_data/OptionMetrics.parquet",
        market_data_path: str = "data/market_data/market_data.csv",
        output_path: str = "storm_data.csv",
        batch_size: int = 1024,
        device: Optional[str] = None
    ):
        self.q_storm_model_path = q_storm_model_path
        self.varphi_model_path = varphi_model_path
        self.varpi_model_path = varpi_model_path
        self.options_data_path = options_data_path
        self.market_data_path = market_data_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


def load_models(config: InferenceConfig) -> Tuple[QStorm, LSTM_Model, VarPi]:
    """
    Load all required models and move them to the appropriate device.

    Args:
        config: Inference configuration object

    Returns:
        Tuple of (q_storm_model, varphi_model, varpi_model)
    """
    logger.info("Loading models...")
    start_time = time.time()

    # Load models
    q_storm = torch.load(config.q_storm_model_path, weights_only=False, map_location=config.device)
    varphi = torch.load(config.varphi_model_path, weights_only=False, map_location=config.device)
    varpi = torch.load(config.varpi_model_path, weights_only=False, map_location=config.device)

    # Move to device and set to eval mode
    q_storm = q_storm.to(config.device).eval()
    varphi = varphi.to(config.device).eval()
    varpi = varpi.to(config.device).eval()

    load_time = time.time() - start_time
    logger.info(f"Models loaded in {load_time:.2f} seconds")

    return q_storm, varphi, varpi


def load_options_data(options_path: str) -> pd.DataFrame:
    """
    Load and preprocess options data.

    Args:
        options_path: Path to options parquet file

    Returns:
        Preprocessed options DataFrame
    """
    logger.info(f"Loading options data from {options_path}...")
    opt_df = pd.read_parquet(options_path, engine='pyarrow')
    opt_df["date"] = pd.to_datetime(opt_df["date"])
    opt_df["exdate"] = pd.to_datetime(opt_df["exdate"])
    opt_df["ticker"] = opt_df["symbol"].str.split(" ").str[0]
    opt_df["days_to_expiration"] = (opt_df["exdate"] - opt_df["date"]).dt.days
    logger.info(f"Loaded {len(opt_df)} option records")
    return opt_df


def get_call_options(opt_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Filter options data for specified tickers and call options.

    Args:
        opt_df: Full options DataFrame
        tickers: List of ticker symbols to filter

    Returns:
        Filtered DataFrame containing only call options for specified tickers
    """
    call_options = opt_df[
        (opt_df["ticker"].isin(tickers)) &
        (opt_df["exercise_style"] == "A") &
        (opt_df["cp_flag"] == "C")
    ].copy()
    logger.info(f"Filtered to {len(call_options)} call options for {len(tickers)} tickers")
    return call_options


def load_market_data(market_data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load market data including prices and risk-free rates.

    Args:
        market_data_path: Path to market data CSV file

    Returns:
        Tuple of (prices DataFrame, risk-free rate Series)
    """
    logger.info(f"Loading market data from {market_data_path}...")
    market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    rf_df = market_data["^IRX"] / 100 / 365
    return market_data, rf_df


def get_prices_and_volatilities(tickers: List[str], options_dir: str = "options_data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price and volatility data for specified tickers.

    Args:
        tickers: List of ticker symbols
        options_dir: Directory containing price CSV files

    Returns:
        Tuple of (prices DataFrame, volatilities DataFrame)
    """
    logger.info(f"Loading prices and volatilities for {len(tickers)} tickers...")
    files = os.listdir(options_dir)
    ticker_data = pd.DataFrame()

    for file in files:
        if any(file.endswith(f"{t}.csv") for t in tickers):
            ticker = file.split(".csv")[0]
            file_path = os.path.join(options_dir, file)
            sub_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            sub_df.rename(columns={"Close": ticker}, inplace=True)
            if ticker_data.empty:
                ticker_data = sub_df
            else:
                ticker_data = ticker_data.join(sub_df, how='outer')

    # Calculate volatilities
    volatilities = ticker_data.pct_change().rolling(window=100).std()
    volatilities = volatilities.dropna()
    prices = ticker_data.loc[volatilities.index]

    logger.info(f"Loaded prices and volatilities for {len(prices.columns)} tickers")
    return prices, volatilities


def get_combined_daily_data(
    call_options_df: pd.DataFrame,
    date: pd.Timestamp,
    rf_rate: float,
    prices: pd.DataFrame,
    volatility: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine options data with normalized stock prices for a specific date.

    Args:
        call_options_df: DataFrame containing all call options
        date: Date to process
        rf_rate: Risk-free rate for the date
        prices: DataFrame of stock prices indexed by date
        volatility: DataFrame of volatilities indexed by date

    Returns:
        DataFrame containing combined data with normalized prices and times
    """
    # Filter options data for the specific date
    daily_options = call_options_df[call_options_df['date'] == date].copy()

    if len(daily_options) == 0:
        raise ValueError(f"No options data found for date {date}")

    # Vectorized operations for better performance
    daily_options["S"] = daily_options["ticker"].map(prices.loc[date])
    daily_options["volatility"] = daily_options["ticker"].map(volatility.loc[date])
    daily_options["K"] = daily_options["strike_price"] / 100
    daily_options["S_prime"] = daily_options["S"] / daily_options["K"]
    daily_options['t_prime'] = daily_options['days_to_expiration']
    daily_options["opt_price"] = (daily_options["best_bid"] + daily_options["best_offer"]) / 2
    daily_options["opt_price_prime"] = daily_options["opt_price"] / daily_options["K"]
    daily_options['rf'] = rf_rate
    daily_options["date"] = date

    # Store original index before resetting
    daily_options["index"] = daily_options.index

    # Keep only relevant columns
    relevant_cols = [
        'date', 'index', 'ticker', 'S_prime', 'K', 't_prime', 'S',
        'volume', 'opt_price_prime', 'opt_price', 'best_bid', 'best_offer',
        'volatility', 'rf'
    ]

    # Only include columns that exist
    available_cols = [col for col in relevant_cols if col in daily_options.columns]
    return daily_options[available_cols].reset_index(drop=True)


def get_varphi_inputs(
    test_datas,
    daily_data: pd.DataFrame,
    date: pd.Timestamp
) -> Tuple[List, pd.DataFrame]:
    """
    Prepare varphi inputs for all unique ticker/time combinations.

    Args:
        test_datas: Test dataset object
        daily_data: Daily options data
        date: Current date

    Returns:
        Tuple of (varphi_inputs list, unique_combs DataFrame with idx column)
    """
    unique_combs = daily_data[["t_prime", "ticker"]].drop_duplicates().copy()
    unique_combs["idx"] = 0
    unique_tickers = unique_combs["ticker"].unique()
    varphi_inputs = []

    for ticker in unique_tickers:
        sub_days = unique_combs[unique_combs["ticker"] == ticker]["t_prime"].values
        for day_till in sub_days:
            if day_till <= 15:
                day_till = 15
            test_datas.set_main_asset(ticker)
            sub_varphi_inps = test_datas.query_by_date(date)
            if day_till != 30:
                sub_varphi_inps = [inp[30-day_till:] for inp in sub_varphi_inps]
            varphi_inputs.append(sub_varphi_inps)
            unique_combs.loc[
                (unique_combs["ticker"] == ticker) &
                (unique_combs["t_prime"] == day_till), "idx"
            ] = len(varphi_inputs) - 1

    return varphi_inputs, unique_combs


def calc_varphi_q_batched(
    varphi: LSTM_Model,
    varphi_inputs: List,
    device: str,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Compute varphi quantiles efficiently.

    Note: Inputs have variable sequence lengths, so we process them individually
    but with optimizations (no_grad, efficient device management).

    Args:
        varphi: VarPhi model
        varphi_inputs: List of varphi input tuples (each with variable length)
        device: Device to run computation on
        batch_size: Not used (kept for API compatibility)

    Returns:
        Stacked tensor of varphi outputs
    """
    varphi_outputs = []

    # Process individually due to variable sequence lengths
    # But use no_grad and efficient device management for speed
    with torch.no_grad():
        for varphi_input in varphi_inputs:
            X, s, Z, _, _ = varphi_input
            X = X.unsqueeze(0).to(device)
            s = s.unsqueeze(0).to(device)
            Z = Z.unsqueeze(0).to(device)

            varphi_q, _ = varphi(X, s, Z)
            varphi_q = varphi_q / 100
            varphi_outputs.append(varphi_q.squeeze().cpu())

    # Stack all outputs
    varphi_outputs = torch.stack(varphi_outputs)
    return varphi_outputs


def calc_varpi_q(
    varpi: VarPi,
    varphi_outputs: torch.Tensor,
    unique_combs: pd.DataFrame,
    device: str
) -> torch.Tensor:
    """
    Compute varpi quantiles from varphi outputs.

    Args:
        varpi: VarPi model
        varphi_outputs: VarPhi output tensor
        unique_combs: DataFrame with t_prime and ticker information
        device: Device to run computation on

    Returns:
        VarPi output tensor
    """
    clipped = np.where(unique_combs["t_prime"].values < 15, 15, unique_combs["t_prime"].values)
    t = unique_combs["t_prime"].values / clipped
    T = clipped / 30

    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
    T_tensor = torch.tensor(T, dtype=torch.float32).unsqueeze(1).to(device)
    varphi_outputs = varphi_outputs.to(device)

    with torch.no_grad():
        varpi_outputs = varpi(varphi_outputs, t_tensor, T_tensor)

    return varpi_outputs


def match_quantiles_with_options(
    options_df: pd.DataFrame,
    quantiles_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match quantile indices with options data.

    Args:
        options_df: Options DataFrame
        quantiles_df: DataFrame with quantile indices

    Returns:
        Merged DataFrame with idx column added
    """
    options_df = options_df.copy()
    options_df['match_key'] = options_df['t_prime'].astype(str) + options_df['ticker']
    quantiles_df['match_key'] = quantiles_df['t_prime'].astype(str) + quantiles_df['ticker']

    merged_df = options_df.merge(
        quantiles_df[['match_key', 'idx']],
        on='match_key',
        how='left'
    )
    merged_df.drop(['match_key'], axis=1, inplace=True)

    return merged_df


def black_scholes_american_call_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray
) -> np.ndarray:
    """
    Vectorized Black-Scholes American call option pricing.
    For American calls without dividends, this equals European call pricing.

    Args:
        S: Stock prices
        K: Strike prices
        T: Time to expiration (in years)
        r: Risk-free rates (annualized)
        sigma: Volatilities (annualized)

    Returns:
        Array of call option prices
    """
    # Avoid division by zero
    T_safe = np.maximum(T, 1e-10)
    sigma_safe = np.maximum(sigma, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return call_price


def inference_storm_batched(
    q_storm: QStorm,
    daily_data: pd.DataFrame,
    varpi_outputs: torch.Tensor,
    device: str
) -> pd.DataFrame:
    """
    Perform batched qStorm inference on all options for a given date.

    Args:
        q_storm: qStorm model
        daily_data: DataFrame containing options data with idx column
        varpi_outputs: VarPi output tensor
        device: Device to run computation on

    Returns:
        DataFrame with added columns: V, V_prime, V_prime_delta, V_delta, BS_V, BS_V_prime
    """
    daily_data = daily_data.copy()

    # Extract data as numpy arrays for vectorized operations
    S_prime_vals = daily_data["S_prime"].values
    t_prime_vals = daily_data["t_prime"].values
    rf_vals = daily_data["rf"].values
    idx_vals = daily_data["idx"].values.astype(int)
    K_vals = daily_data["K"].values
    S_vals = daily_data["S"].values
    volatility_vals = daily_data["volatility"].values

    # Convert to tensors for batched processing
    S_prime_tensor = torch.tensor(S_prime_vals, dtype=torch.float32).unsqueeze(
        1).to(device).requires_grad_(True)
    t_prime_tensor = torch.tensor(t_prime_vals, dtype=torch.float32).unsqueeze(1).to(device)
    rf_tensor = torch.tensor(rf_vals, dtype=torch.float32).unsqueeze(1).to(device)

    # Get corresponding varpi quantiles
    varpi_q_batch = varpi_outputs[idx_vals].to(device)

    # Batched forward pass
    with torch.no_grad():
        V_prime_tensor = q_storm(S_prime_tensor, t_prime_tensor, rf_tensor, varpi_q_batch)

    # Compute deltas in batch
    V_prime_tensor.requires_grad_(True)
    S_prime_tensor.requires_grad_(True)

    # Re-compute with gradients enabled for delta calculation
    V_prime_with_grad = q_storm(S_prime_tensor, t_prime_tensor, rf_tensor, varpi_q_batch)

    # Compute gradients for all options at once
    grad_outputs = torch.ones_like(V_prime_with_grad)
    deltas_tensor = torch.autograd.grad(
        outputs=V_prime_with_grad,
        inputs=S_prime_tensor,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False
    )[0]

    # Convert to numpy
    V_prime_vals = V_prime_tensor.detach().cpu().numpy().flatten()
    deltas_vals = deltas_tensor.detach().cpu().numpy().flatten()

    # Vectorized Black-Scholes computation
    T_years = t_prime_vals / 365.0
    r_annual = rf_vals * 365.0
    sigma_annual = volatility_vals * np.sqrt(365.0)

    BS_V_vals = black_scholes_american_call_vectorized(
        S_vals, K_vals, T_years, r_annual, sigma_annual
    )

    # Compute final values
    V_vals = V_prime_vals * K_vals
    V_delta_vals = deltas_vals * K_vals

    # Add results to DataFrame
    daily_data["V"] = V_vals
    daily_data["V_prime"] = V_prime_vals
    daily_data["V_prime_delta"] = deltas_vals
    daily_data["V_delta"] = V_delta_vals
    daily_data["BS_V"] = np.round(BS_V_vals, 6)
    daily_data["BS_V_prime"] = daily_data["BS_V"] / daily_data["K"]

    return daily_data


def process_dates(
    q_storm: QStorm,
    varphi: LSTM_Model,
    varpi: VarPi,
    call_opts: pd.DataFrame,
    daily_rf: pd.Series,
    prices: pd.DataFrame,
    volatilities: pd.DataFrame,
    test_datas,
    config: InferenceConfig
) -> pd.DataFrame:
    """
    Process all dates and perform inference.

    Args:
        q_storm: qStorm model
        varphi: VarPhi model
        varpi: VarPi model
        call_opts: Call options DataFrame
        daily_rf: Risk-free rate Series
        prices: Prices DataFrame
        volatilities: Volatilities DataFrame
        test_datas: Test dataset object
        config: Inference configuration

    Returns:
        Combined DataFrame with all inference results
    """
    device = config.device
    dates = call_opts["date"].unique()
    results = []

    logger.info(f"Processing {len(dates)} dates...")

    for date in tqdm(dates, desc="Processing dates"):
        try:
            # Get risk-free rate for date
            try:
                rf = daily_rf.loc[date]
                if pd.isna(rf):
                    rf = daily_rf.iloc[-1]  # Use last available rate
            except (KeyError, AttributeError):
                rf = daily_rf.iloc[-1] if len(daily_rf) > 0 else 0.0

            # Get daily data
            daily_data = get_combined_daily_data(call_opts, date, rf, prices, volatilities)

            # Get varphi inputs
            varphi_inputs, unique_combs = get_varphi_inputs(test_datas, daily_data, date)

            # Compute varphi quantiles (batched)
            varphi_q = calc_varphi_q_batched(varphi, varphi_inputs, device, config.batch_size)

            # Compute varpi quantiles
            varpi_q = calc_varpi_q(varpi, varphi_q, unique_combs, device)

            # Match quantiles with options
            daily_data = match_quantiles_with_options(daily_data, unique_combs)

            # Perform qStorm inference (batched)
            daily_data = inference_storm_batched(q_storm, daily_data, varpi_q, device)

            results.append(daily_data)

        except Exception as e:
            logger.error(f"Error processing date {date}: {e}", exc_info=True)
            continue

    # Concatenate all results at once (much faster than repeated concat)
    if results:
        final_df = pd.concat(results, ignore_index=True)
        logger.info(f"Processed {len(final_df)} option records across {len(dates)} dates")
        return final_df
    else:
        logger.warning("No results to concatenate")
        return pd.DataFrame()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run qStorm inference on options data')
    parser.add_argument('--tickers', nargs='+', default=["XOM", "NOK", "MCD", "NKE", "PFE", "BB", "GME", "JPM"],
                        help='List of ticker symbols to process')
    parser.add_argument('--q-storm-model', default='models/q_storm.pth',
                        help='Path to qStorm model')
    parser.add_argument('--varphi-model', default='models/varphi.pth',
                        help='Path to VarPhi model')
    parser.add_argument('--varpi-model', default='models/varpi.pth',
                        help='Path to VarPi model')
    parser.add_argument('--output', default='storm_data.csv',
                        help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for varphi processing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), auto-detected if not specified')

    args = parser.parse_args()

    # Create configuration
    config = InferenceConfig(
        q_storm_model_path=args.q_storm_model,
        varphi_model_path=args.varphi_model,
        varpi_model_path=args.varpi_model,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device
    )

    logger.info("=" * 60)
    logger.info("qStorm Inference Script")
    logger.info("=" * 60)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Output: {config.output_path}")

    start_time = time.time()

    # Load models
    q_storm, varphi, varpi = load_models(config)

    # Load data
    opt_df = load_options_data(config.options_data_path)
    call_opts = get_call_options(opt_df, args.tickers)
    _, daily_rf = load_market_data(config.market_data_path)
    prices, volatilities = get_prices_and_volatilities(args.tickers)
    test_datas = get_test_dataset(219, "2020-01-01", "2025-01-01", 30, True)

    # Process all dates
    storm_data = process_dates(
        q_storm, varphi, varpi, call_opts, daily_rf, prices, volatilities,
        test_datas, config
    )

    # Save results
    if not storm_data.empty:
        storm_data.to_csv(config.output_path, index=False)
        logger.info(f"Results saved to {config.output_path}")
        logger.info(f"Total records: {len(storm_data)}")
    else:
        logger.warning("No data to save")

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Inference completed in {total_time:.2f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
