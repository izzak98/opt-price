import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna


class Position:
    def __init__(self, entry_date, ticker, K, entry_price, t_prime, stop_loss, take_profit, model_price):
        self.entry_date = entry_date
        self.ticker = ticker
        self.K = K
        self.entry_price = entry_price  # Price paid (normalized)
        self.current_price = entry_price
        self.days_to_expiry = t_prime
        self.stop_loss = stop_loss  # As decimal (e.g., 0.05 for 5%)
        self.take_profit = take_profit  # As decimal
        self.model_entry_price = model_price
        self.model_exit_price = model_price
        self.status = 'open'


class TradingSimulator:
    def __init__(self, data, params):
        self.data = data.sort_values('date')
        self.params = params
        self.current_date = None
        self.positions = []
        self.closed_positions = []

        # Parameters
        self.commission = params.get('commission', 0.001)  # 0.1%
        self.slippage = params.get('slippage', 0.0005)  # 0.05%
        self.stop_loss = params.get('stop_loss', 0.10)
        self.take_profit = params.get('take_profit', 0.20)
        self.signal_distance = params.get('signal_distance', 0.001)
        self.close_signal_distance = params.get('close_signal_distance', 0.0005)
        self.capital_per_trade = params.get('capital_per_trade', 1000)
        self.selected_asset = params.get('asset', 'JPM')
        self.n_days = params.get('n_days', 30)
        self.max_positions = params.get('max_positions', 10)

    def run(self):
        unique_dates = self.data['date'].unique()

        for i, date in enumerate(sorted(unique_dates)):
            self.current_date = date
            self.process_existing_positions(date)
            if len(self.positions) < self.max_positions:
                self.check_new_signals(date)
            if i == self.n_days:
                break

    def process_existing_positions(self, current_date):
        for pos in list(self.positions):  # Iterate over copy for safe removal
            # Get current day's data for this option
            # Update days to expiry
            pos.days_to_expiry -= 1
            current_data = self.data[(self.data['date'] == current_date) &
                                     (self.data['ticker'] == pos.ticker) &
                                     (self.data['K'] == pos.K) &
                                     (self.data['t_prime'] == pos.days_to_expiry)]

            # Check expiration
            if pos.days_to_expiry <= 0:
                self.close_position(pos, current_date, reason='expired')
                continue

            # Update current price if data available
            if not current_data.empty:
                opt_price = current_data['opt_price'].item()
                pos.current_price = opt_price  # Update to current normalized price

                # Check stop loss/take profit
                pct_change = (pos.current_price - pos.entry_price)/pos.entry_price

                if pct_change <= -self.stop_loss:
                    self.close_position(pos, current_date, reason='stop_loss')
                elif pct_change >= self.take_profit:
                    self.close_position(pos, current_date, reason='take_profit')
                else:
                    # Check signal reversal exit
                    V_prime = current_data['V_prime'].item()
                    signal_ratio = V_prime
                    market_ratio = current_data['opt_price_prime'].item()
                    signal = (signal_ratio - market_ratio) / market_ratio

                    if signal < self.close_signal_distance and current_data['volume'].item() > 0:
                        self.close_position(pos, current_date, reason='signal_reversal')

    def close_position(self, position, exit_date, reason):
        # Get exit price data
        exit_data = self.data[(self.data['date'] == exit_date) &
                              (self.data['ticker'] == position.ticker) &
                              (self.data['K'] == position.K) &
                              (self.data['t_prime'] == position.days_to_expiry)]

        if exit_data.empty:  # Handle missing data at expiry
            exit_price = 0  # Assume worthless if no data
            model_exit_price = 0
        else:
            # Calculate exit price with slippage
            exit_price = exit_data['best_bid'].item()
            model_exit_price = exit_data['V'].item()

        # Calculate returns
        pnl = (exit_price - position.entry_price) * self.capital_per_trade
        pnl -= self.commission * self.capital_per_trade * 2  # Entry and exit commissions

        # Record closed position
        self.closed_positions.append({
            'entry_date': position.entry_date,
            'exit_date': exit_date,
            'ticker': position.ticker,
            'pnl': pnl,
            'reason': reason,
            'days_held': (exit_date - position.entry_date).days,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'K': position.K,
            'model_entry_price': position.model_entry_price,
            'model_exit_price': model_exit_price
        })

        self.positions.remove(position)

    def check_new_signals(self, current_date):
        day_data = self.data[(self.data['date'] == current_date) &
                             (self.data['ticker'] == self.selected_asset)]
        day_data = day_data.sort_values('V_prime', ascending=False)
        for _, row in day_data.iterrows():
            if len(self.positions) >= self.max_positions:
                break
            signal_ratio = row['V_prime']
            market_ratio = row['opt_price_prime']
            signal = (signal_ratio - market_ratio)/market_ratio

            # Entry condition
            if signal > self.signal_distance:
                # Calculate entry price with slippage
                entry_price = row['best_offer'] * (1 + self.slippage)

                new_pos = Position(
                    entry_date=current_date,
                    ticker=row['ticker'],
                    K=row['K'],
                    entry_price=entry_price,
                    t_prime=row['t_prime'],
                    stop_loss=self.stop_loss,
                    take_profit=self.take_profit,
                    model_price=row['V']
                )

                self.positions.append(new_pos)

    def get_performance_report(self):
        df = pd.DataFrame(self.closed_positions)
        df['cumulative_pnl'] = df['pnl'].cumsum()
        return df


# Example Usage
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('storm_data.csv', parse_dates=['date'])

    # def objective(trial):
    #     params = {
    #         'commission': 0.001,  # 0.1% per trade
    #         'slippage': 0.0005,
    #         'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.15, step=0.01),
    #         'take_profit': trial.suggest_float('take_profit', 0.01, 0.25, step=0.01),
    #         'signal_distance': trial.suggest_float('signal_distance', 0.001, 0.1, step=0.001),
    #         'close_signal_distance': trial.suggest_float('close_signal_distance', 0.001, 0.1, step=0.001),
    #         'max_positions': trial.suggest_int('max_positions', 5, 50),
    #         'capital_per_trade': 100,
    #         'n_days': 30,
    #         'asset': 'GME'
    #     }

    #     simulator = TradingSimulator(data, params)
    #     simulator.run()
    #     try:
    #         results = simulator.get_performance_report()
    #     except Exception as e:
    #         return float("-inf")
    #     return results['pnl'].sum()

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100, n_jobs=1)
    # print(study.best_params)
    params = {
        'commission': 0.01,  # 0.1% per trade
        'slippage': 0.001,
        'stop_loss': 0.1,  # 2% stop loss
        'take_profit': 0.04,  # 4% take profit
        'signal_distance': 1,  # 2% distance between signals
        'close_signal_distance': 0.01,  # 1% distance between close signals
        'max_positions': 5,
        'capital_per_trade': 100,
        'n_days': 60,
        'asset': 'JPM'
    }

    simulator = TradingSimulator(data, params)
    simulator.run()
    print(simulator.get_performance_report())
