import pandas as pd
import numpy as np
from typing import Optional


class OptionsBacktest:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades = []

    def calculate_mispricing(self) -> pd.DataFrame:
        """Calculate mispricing based on model V vs market price"""
        self.data['mispricing'] = (self.data['V'] * self.data['K']) - self.data['opt_price']
        self.data['mispricing_pct'] = self.data['mispricing'] / self.data['opt_price']
        return self.data

    def generate_signals(self,
                         mispricing_threshold: float = 0.15,
                         min_volume: int = 10,
                         max_position_size: float = 0.1) -> pd.DataFrame:
        """Generate trading signals based on mispricing"""
        # Only consider options with sufficient volume
        self.data['signal'] = 0
        mask = (self.data['volume'] >= min_volume)

        # Buy signal when model price is significantly higher than market price
        self.data.loc[mask & (self.data['mispricing_pct'] > mispricing_threshold), 'signal'] = 1

        # Calculate position sizes based on confidence (mispricing magnitude)
        self.data['position_size'] = np.minimum(
            self.data['mispricing_pct'] * max_position_size,
            max_position_size
        )

        return self.data

    def execute_trades(self, transaction_cost: float = 0.01):
        """Simulate execution of trades based on signals"""
        for _, row in self.data.iterrows():
            if row['signal'] == 1:
                # Calculate position size in dollars
                position_dollars = self.current_capital * row['position_size']
                # Each contract is for 100 shares
                num_contracts = int(position_dollars / (row['opt_price'] * 100))

                if num_contracts > 0:
                    # Account for transaction costs
                    cost = (num_contracts * row['opt_price'] * 100) * (1 + transaction_cost)

                    if cost <= self.current_capital:
                        trade = {
                            'date': row['date'],
                            'ticker': row['ticker'],
                            'strike': row['K'],
                            'contracts': num_contracts,
                            'entry_price': row['opt_price'],
                            'cost': cost,
                            'expected_return': row['opt_returns'],
                            'mispricing': row['mispricing'],
                            'mispricing_pct': row['mispricing_pct']
                        }

                        self.trades.append(trade)
                        self.current_capital -= cost

    def calculate_performance(self) -> dict:
        """Calculate strategy performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}

        trades_df = pd.DataFrame(self.trades)

        # Calculate returns
        total_investment = sum(trade['cost'] for trade in self.trades)
        total_return = sum(trade['cost'] * trade['expected_return'] for trade in self.trades)

        # Performance metrics
        performance = {
            'total_trades': len(self.trades),
            'total_investment': total_investment,
            'final_capital': self.current_capital + total_return,
            'total_return': total_return,
            'return_pct': (total_return / total_investment) * 100 if total_investment > 0 else 0,
            'avg_mispricing': trades_df['mispricing'].mean(),
            'avg_mispricing_pct': trades_df['mispricing_pct'].mean(),
            'trades_by_ticker': trades_df['ticker'].value_counts().to_dict()
        }

        return performance


def run_backtest(data: pd.DataFrame,
                 initial_capital: float = 100000,
                 mispricing_threshold: float = 0.15,
                 min_volume: int = 10,
                 max_position_size: float = 0.1,
                 transaction_cost: float = 0.01) -> dict:
    """
    Run complete backtest with given parameters

    Args:
        data: DataFrame containing options data
        initial_capital: Starting capital for the strategy
        mispricing_threshold: Minimum mispricing percentage to trigger a trade
        min_volume: Minimum volume required for a trade
        max_position_size: Maximum position size as percentage of capital
        transaction_cost: Transaction cost as percentage of trade value

    Returns:
        Dictionary containing backtest results and performance metrics
    """
    backtest = OptionsBacktest(data, initial_capital)

    # Execute strategy steps
    backtest.calculate_mispricing()
    backtest.generate_signals(mispricing_threshold, min_volume, max_position_size)
    backtest.execute_trades(transaction_cost)

    # Calculate and return performance
    return backtest.calculate_performance()


# Example usage:
if __name__ == "__main__":
    # Sample parameters
    params = {
        'initial_capital': 100000,
        'mispricing_threshold': 0.01,
        'min_volume': 10,
        'max_position_size': 0.1,
        'transaction_cost': 0.01
    }
    data = pd.read_csv('storm_data.csv')

    # Run backtest with sample parameters
    results = run_backtest(data, **params)
    print("\nBacktest Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
