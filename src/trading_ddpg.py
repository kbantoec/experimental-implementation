# Standard library imports
from time import time
from pathlib import Path

# Related third party imports
import gym
from gym.envs.registration import register
import numpy as np
import pandas as pd
from numpy.core import ndarray
from pandas.core.frame import DataFrame, Series
from rl.agents import DDPGAgent

# Local application/library specific imports
from src.envs.utils import track_results
from src.envs import EquallyWeightedPortfolio, CompositePortfolio
from src.preprocessing import PricesProcessor
from src.ddpg import Agent

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'
# Todo: How to use Tensorboard?
# Todo: Create an RL agent
# Todo: replicate https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530
# Todo: Check this out "Using TensorTrade for Making a Simple Trading Algorithm"
#  Link -> https://levelup.gitconnected.com/using-tensortrade-for-making-a-simple-trading-algorithm-6fad4d9bc79c
# Todo: read https://keras-rl.readthedocs.io/en/latest/ and https://github.com/keras-rl/keras-rl/tree/master/rl/agents
# Deep Reinforcement Learning Tutorial for Python in 20 Minutes: https://www.youtube.com/watch?v=cO5g5qLrLSo
# https://github.com/AI4Finance-LLC/FinRL-Library


if __name__ == '__main__':
    # Constants
    INITIAL_CASH: int = 1_000
    TRADING_PERIODS: int = 5  # Episode length
    TRANSACTION_COST: float = 1e-3
    MAX_EPISODES: int = 2_000
    TRAINING: bool = True

    # Gym environment registration
    register(id='TradingEnv-v0',
             entry_point='envs.trading:TradingEnv',
             max_episode_steps=TRADING_PERIODS)

    # TODO: compose the dataset with different building blocks using '+'
    # dataset = features + rf + inflation
    # data: ndarray = np.load(DATA_PATH / 'tensor.npy')

    # Load risk-free rates
    store = pd.HDFStore(DATA_PATH / 'DGS1MO.h5')
    annual_rf = pd.DataFrame(store.get('rf'))
    store.close()

    # Load macroeconomic indicators
    store = pd.HDFStore(DATA_PATH / 'macro.h5')
    indicators = pd.DataFrame(store.get('USARECDM'))
    store.close()

    dataset = PricesProcessor(DATA_PATH,
                              'sp500_closefull.csv',
                              rf=annual_rf.loc[:, ['DGS1MO_annual']],
                              macroindicators=indicators)
    dataset.load_tensor(DATA_PATH / 'tensor.npy')

    ew_portfolio = EquallyWeightedPortfolio(INITIAL_CASH,
                                            TRANSACTION_COST,
                                            TRADING_PERIODS,
                                            num_risky_assets=dataset.num_assets)

    portfolios = CompositePortfolio(ew_portfolio)

    # mkt_portfolio = MarketPortfolio(...)
    # portfolios = ew_portfolio + mkt_portfolio  # CompositePortfolio object

    # Environment instantiation
    trading_environment = gym.make('TradingEnv-v0',
                                   initial_cash=INITIAL_CASH,
                                   transaction_cost=TRANSACTION_COST,
                                   dataset=dataset,
                                   baseline_portfolios=portfolios.portfolios,
                                   verbose=True)
    trading_environment.seed(42)

    agent = Agent(input_dims=trading_environment.observation_space.shape,  # (480, 10)
                  env=trading_environment,
                  max_size=1_000,
                  n_actions=trading_environment.action_space.shape[0])  # (480,)

    print(agent.memory.get_size_per_attr())

    #     File "C:/Users/YBant/Documents/projects/drl-thesis/src/trading_ddpg.py", line 78, in <module>
    #         n_actions=trading_environment.action_space.shape[0])
    #     File "C:\Users\YBant\Documents\projects\drl-thesis\src\ddpg\ddpg.py", line 51, in __init__
    #         self.memory = ReplayBuffer(max_size, input_dims, n_actions)
    #     File "C:\Users\YBant\Documents\projects\drl-thesis\src\ddpg\buffer.py", line 28, in __init__
    #         self.new_state_memory: ndarray = np.zeros((self.mem_size, *input_shape))
    # MemoryError: Unable to allocate 35.8 GiB for an array with shape (1000000, 480, 10) and data type float64

    arr = np.zeros(dataset.num_assets)
    arr[3] = 0.2
    arr[5] = 0.3

    state_dim: int = trading_environment.observation_space.shape[0]
    start: float = time()
    results: list = []
    episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

    for episode in range(1, MAX_EPISODES + 1):
        # Todo: Check if it works with the [1...0] initial positions
        this_state: ndarray = trading_environment.reset()
        # print(trading_environment.simulator.trader_portfolio.positions)
        for episode_step in range(MAX_EPISODES):
            action = agent.choose_action(this_state.reshape(-1, state_dim), TRAINING)
            next_state, reward, done, _ = trading_environment.step(action)

            agent.memorize_transition(this_state,
                                     action,
                                     reward['trader'],
                                     next_state,
                                     0.0 if done else 1.0)
            if agent.train:
                agent.experience_replay()
            if done:
                break
            this_state = next_state

        result: DataFrame = trading_environment.env.simulator.result()
        final = result.iloc[-1]

        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        market_nav = final.market_nav
        market_navs.append(market_nav)

        diff = nav - market_nav
        diffs.append(diff)
        if episode % 10 == 0:
            track_results(episode, np.mean(navs[-100:]), np.mean(navs[-10:]),
                          np.mean(market_navs[-100:]), np.mean(market_navs[-10:]),
                          np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
                          time() - start, agent.epsilon)
        if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
            print(result.tail())
            break

    trading_environment.close()
