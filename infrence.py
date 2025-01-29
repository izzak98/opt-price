import json
import warnings
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from q_storm.qStorm import QStorm
from varpi.tain_varpi import VarPi
from varpi.QLSTM import LSTM_Model
from utils.data_utils import get_test_dataset
from scipy.stats import norm

warnings.filterwarnings("ignore")

accelerator = Accelerator()


def load_varphi_inputs():
    return get_test_dataset(219, "2020-01-01", "2025-01-01", 30, True)


def load_options_data():
    opt_df = pd.read_parquet('options_data/OptionMetrics.parquet',
                             engine='pyarrow')  # or engine='fastparquet'
    opt_df["date"] = pd.to_datetime(opt_df["date"])
    opt_df["exdate"] = pd.to_datetime(opt_df["exdate"])
    opt_df["ticker"] = opt_df["symbol"].str.split(" ").str[0]
    opt_df["days_to_expiration"] = (opt_df["exdate"] - opt_df["date"]).dt.days
    return opt_df


def get_rf():
    market_data = pd.read_csv("data/market_data/market_data.csv", index_col=0, parse_dates=True)
    rf_df = market_data["^IRX"]/100/365
    return rf_df


def get_call_options(tickers: list[str]):
    opt_df = load_options_data()
    opt_df = opt_df[(opt_df["ticker"].isin(tickers)) & (opt_df["exercise_style"] == "A")]
    call_options = opt_df[opt_df["cp_flag"] == "C"]
    del opt_df
    return call_options


def get_prices(tickers):
    files = os.listdir("options_data")
    ticker_data = pd.DataFrame()
    for file in files:
        if any(file.endswith(f"{t}.csv") for t in tickers):
            ticker = file.split(".csv")[0]
            sub_df = pd.read_csv(f"options_data/{file}", index_col=0, parse_dates=True)
            sub_df.rename(columns={"Close": f"{ticker}"}, inplace=True)
            ticker_data = ticker_data.join(sub_df, how='right')
    prices = ticker_data
    volatilities = ticker_data.pct_change().rolling(window=100).std()
    volatilities = volatilities.dropna()
    prices = prices.loc[volatilities.index]
    return prices, volatilities


def get_combined_daily_data(call_options_df, date, rf_rate, prices, volatility):
    """
    Combine options data with normalized stock prices for a specific date

    Parameters:
    -----------
    options_df: pandas DataFrame
        Options data containing all options fields
    sprime_series: pandas Series
        Normalized stock prices (S_prime) indexed by ticker
    date: str or datetime
        The specific date to process
    rf_rate: float, optional
        Risk-free rate for the given date

    Returns:
    --------
    DataFrame containing combined data with:
        S_prime: normalized stock price from sprime_series
        K_prime: normalized strike prices
        T_prime: normalized time
        rf: risk-free rate
    """
    # Filter options data for the specific date
    daily_options = call_options_df[call_options_df['date'] == date].copy()

    if len(daily_options) == 0:
        raise ValueError(f"No options data found for date {date}")

    # Get S_prime for each ticker
    daily_options["S"] = daily_options["ticker"].map(prices.loc[date])
    daily_options["volatility"] = daily_options["ticker"].map(volatility.loc[date])

    daily_options["K"] = daily_options["strike_price"]/100

    # Normalize stock price
    daily_options["S_prime"] = daily_options["S"]/daily_options["K"]
    # Normalize time (you mentioned this is usually just one)
    daily_options['t_prime'] = daily_options['days_to_expiration']

    daily_options["opt_price"] = (daily_options["best_bid"] + daily_options["best_offer"])/2
    daily_options["opt_price_prime"] = daily_options["opt_price"] / daily_options["K"]
    daily_options["index"] = daily_options.index

    daily_options['rf'] = rf_rate

    daily_options["date"] = date

    # Keep only relevant columns
    relevant_cols = ['date', 'index', 'ticker', 'S_prime', 'K', 't_prime', 'S',
                     'volume', 'opt_price_prime', 'opt_price', 'best_bid', 'best_offer', 'volatility', 'rf']

    return daily_options[relevant_cols]


def get_varphi_inps(test_datas, daily_data, date):
    unique_combs = daily_data[["t_prime", "ticker"]].drop_duplicates()
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
                (unique_combs["t_prime"] == day_till), "idx"] = len(varphi_inputs) - 1
    return varphi_inputs, unique_combs


def calc_varphi_q(varphi, varphi_inputs):
    varphi_outputs = []
    for varphi_input in varphi_inputs:
        X, _, Z, _, _ = varphi_input
        X = X.unsqueeze(0)
        Z = Z.unsqueeze(0)
        varphi_q, _ = varphi(X, _, Z)
        varphi_q = varphi_q/100
        varphi_q = varphi_q.squeeze()
        varphi_outputs.append(varphi_q)
    varphi_outputs = torch.stack(varphi_outputs)
    return varphi_outputs


def calc_varpi_q(varpi, varphi_outputs, unique_combs):
    clipped = np.where(unique_combs["t_prime"].values < 15, 15, unique_combs["t_prime"].values)
    t = unique_combs["t_prime"] / clipped
    T = clipped/30
    t = torch.tensor(t.values).float().unsqueeze(1)
    T = torch.tensor(T).float().unsqueeze(1)

    device = accelerator.device
    t = t.to(device)
    T = T.to(device)

    varpi_outputs = varpi(varphi_outputs, t, T)
    return varpi_outputs


def match_quantiles_with_options(options_df, quantiles_df):
    options_df = options_df.copy()

    # Create matching key in both dataframes
    options_df['match_key'] = options_df['t_prime'].astype(str) + options_df['ticker']
    quantiles_df['match_key'] = quantiles_df['t_prime'].astype(
        str) + quantiles_df['ticker']

    # Merge the dataframes
    merged_df = options_df.merge(
        quantiles_df[['match_key', 'idx']],
        on='match_key',
        how='left'
    )

    # Clean up
    merged_df.drop(['match_key'], axis=1, inplace=True)

    return merged_df


def black_scholes_american_call(S, K, T, r, sigma):
    # No dividends: American call = European call
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price


def infrence_storm(q_storm, daily_data, varpi_outputs):
    device = accelerator.device
    Vs = []
    V_primes = []
    BS_Vs = []
    rf = daily_data["rf"].iloc[0].item()
    rf_torch = torch.tensor([[rf]]).float().to(device)
    for idx, row in daily_data.iterrows():
        S_prime = row["S_prime"]
        t_prime = row["t_prime"]
        idx = row["idx"]
        varpi_q = varpi_outputs[idx].unsqueeze(0)
        S_prime = torch.tensor([[S_prime]]).float().to(device)
        t_prime = torch.tensor([[t_prime]]).float().to(device)
        V_prime = q_storm(S_prime, t_prime, rf_torch, varpi_q)
        K = row["K"]
        BS_V = black_scholes_american_call(S=row["S"],
                                           K=K,
                                           T=row["t_prime"]/365,
                                           r=rf*365,
                                           sigma=row["volatility"]*np.sqrt(365)
                                           )
        V = V_prime * K
        Vs.append(V.item())
        V_primes.append(V_prime.item())
        BS_Vs.append(BS_V)
    daily_data["V"] = Vs
    daily_data["V_prime"] = V_primes
    daily_data["BS_V"] = np.round(BS_Vs, 6)
    daily_data["BS_V_prime"] = daily_data["BS_V"]/daily_data["K"]
    return daily_data


if __name__ == "__main__":
    q_storm = torch.load('models/q_storm.pth')
    varphi = torch.load('models/varphi.pth')
    varpi = torch.load('models/varpi.pth')

    test_datas = load_varphi_inputs()
    call_opts = get_call_options(["GME", "BB", "JPM"])
    daily_rf = get_rf()
    prices, volatilities = get_prices(["GME", "BB", "JPM"])
    storm_data = pd.DataFrame()
    dates = call_opts["date"].unique()
    for date in tqdm(dates, desc="inferencing days"):
        try:
            rf = daily_rf.loc[date].item()
        except (AttributeError, KeyError):
            rf = rf
        daily_data = get_combined_daily_data(call_opts, date, rf, prices, volatilities)
        varphi_inputs, unique_combs = get_varphi_inps(test_datas, daily_data, date)
        varphi_q = calc_varphi_q(varphi, varphi_inputs)
        varpi_q = calc_varpi_q(varpi, varphi_q, unique_combs)
        daily_data = match_quantiles_with_options(daily_data, unique_combs)
        daily_data = infrence_storm(q_storm, daily_data, varpi_q)
        storm_data = pd.concat((storm_data, daily_data), axis=0)
    storm_data.to_csv("storm_data.csv")
    print(daily_data)
