import gym
from gym.envs.registration import register
import pandas as pd
from pandas.core.frame import DataFrame
from envs import Portfolio2 as Portfolio
from pathlib import Path

DIR_PATH = Path(__file__).resolve().parent
DATA_PATH = DIR_PATH / 'data'

if __name__ == '__main__':
    # Constants
    INITIAL_CASH: int = 1_000
    TRADING_PERIODS: int = 5  # Episode length
    TRANSACTION_COST: float = 1e-3

    # Gym environment registration
    register(id='TradingEnv-v0',
             entry_point='envs:TradingEnv',
             max_episode_steps=TRADING_PERIODS)

    # Importing data
    data = pd.read_csv(DATA_PATH / 'sp500_closefull.csv', parse_dates=['Date'], index_col=['Date'])
    ptf_data: DataFrame = data[['AAPL', 'MSFT']]

    # Create portfolios
    ptf = Portfolio(INITIAL_CASH, ptf_data, TRADING_PERIODS, TRANSACTION_COST)

    # Environment instantiation
    env = gym.make('TradingEnv-v0', trader_portfolio=ptf, verbose=True)
    print(env.action_space.sample())
    print(env.observation_space.sample())
    env.reset()
    s1 = env.step([0.3, 0.7])
    s2 = env.step([0.2, 0.5])
