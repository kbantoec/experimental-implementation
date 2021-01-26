from __future__ import annotations  # To allow parameterized type hints

import gym
from gym import spaces
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Type hints
from numpy.core import ndarray
from pandas.core.frame import DataFrame, Series
from typing import Optional, Union

# Local imports
from src.envs.utils import check_type
from src.preprocessing.data_processors import DataProcessor, PricesProcessor


class Portfolio(ABC):
    """
    Portfolio.

    :ivar n: Number of assets, including the risk-free asset.
    :ivar offset: The chosen starting point in the dataset.
    :ivar initial_weights: Initial weights is the initial
    portfolio asset allocation. It is defined as:
      -> 100% in the bank account (i.e., the risk-free asset),
      -> 0% in every risky asset.
    :ivar _weights: The portfolio weights. These are the
    "actions" taken by the agent. Also called "positions".
    :ivar positions_ante: The ex-ante weights. That is, the
    weights at t-1 or just before the period rebalancing.
    :ivar unrolled_returns: The weighted-returns of each
    asset in which the agent is long.
    :ivar cash_positions: The monetary value of each
    position the agent is in.
    position the agent is in.
    :ivar cash_positions_ante: The ex-ante cash positions.
    That is, the cash positions at t-1.
    :ivar value: The monetary value of the portfolio. That
    is, the sum of the cash positions at time t.
    :ivar value_ante: The ex-ante portfolio monetary value.
    :ivar rewards: Rewards are used to compute the total
    return. Thus, they are the portfolio returns.
    :ivar nav: The Net Asset Value (NAV).
    :ivar nav_ante: The ex-ante NAV.
    :ivar turnover_per_risky_asset: How much of each asset
    has been traded, expect the risk-free asset. Including
    the latter would double the turnover of the period,
    which is wrong. The rationale is that it costs nothing
    to deposit of withdraw cash from the bank account.
    :ivar turnover: The total amount of traded assets.
    :ivar trading_costs: The monetary value of the turnover.
    That is, it is the cost of trading, making transactions.
    """
    def __init__(self,
                 initial_cash: Union[int, float],
                 transaction_cost: Union[int, float],
                 trading_periods: int,
                 num_risky_assets: int,
                 name: Optional[str] = None):
        # Raise TypeError messages for debugging
        check_type(initial_cash, [int, float])
        check_type(transaction_cost, [int, float])
        check_type(trading_periods, [int])
        check_type(num_risky_assets, [int])
        check_type(name, [str, type(None)])

        # Constant attributes
        self.initial_cash: Union[int, float] = initial_cash
        self.tau: Union[int, float] = transaction_cost
        self.trading_periods: int = trading_periods
        self.n: int = num_risky_assets + 1  # + 1 is for the risk-free asset
        self.name: str = hex(id(self)) if name is None else name

        # Changing attributes
        self.offset: Optional[int] = None

        self.initial_weights: ndarray = np.append(1, np.repeat(0, self.n - 1))
        self._weights: ndarray = np.zeros((self.trading_periods, self.n))
        self.positions_ante: ndarray = np.zeros_like(self._weights)
        self.positions_ante[0][0] = 1

        self.unrolled_returns: ndarray = np.zeros_like(self._weights)

        self.cash_positions_ante: ndarray = np.zeros_like(self.positions_ante)
        self.cash_positions_ante[0][0] = self.initial_cash
        self.cash_positions: ndarray = np.zeros_like(self._weights)

        self.value_ante: ndarray = np.sum(self.cash_positions_ante, axis=1)
        self.value: ndarray = np.zeros_like(self.value_ante)

        self.rewards: ndarray = np.zeros(self.trading_periods)  # = Portfolio returns
        self.nav: ndarray = np.ones(self.trading_periods)  # = Portfolio value of $1
        self.nav_ante: ndarray = np.ones(self.trading_periods)
        self.turnover_per_risky_asset: ndarray = np.zeros((self.trading_periods, self.n - 1))
        self.turnover: ndarray = np.zeros(self.trading_periods)
        self.trading_costs: ndarray = np.zeros(self.trading_periods)  # Monetary value

        self._verbose: bool = False

    @abstractmethod
    def step(self, *args):
        pass

    def reset(self, offset: int):
        self.offset = offset

        self._weights.fill(0)
        self.positions_ante.fill(0)
        self.positions_ante[0][0] = 1

        self.unrolled_returns.fill(0)

        self.cash_positions_ante.fill(0)
        self.cash_positions_ante[0][0] = self.initial_cash
        self.cash_positions.fill(0)

        # The portfolio value is the sum of the cash positions in each asset
        self.value_ante = np.sum(self.cash_positions_ante, axis=1)
        self.value.fill(0)

        self.rewards.fill(0)
        self.nav.fill(1)
        self.nav_ante.fill(1)
        self.turnover_per_risky_asset.fill(0)
        self.turnover.fill(0)
        self.trading_costs.fill(0)

        self._verbose = False

    @property
    def positions(self):
        """The actions the agent takes is the portion
        of money to allocate in each asset. Thus, the
        weights or the positions."""
        return self._weights

    def __add__(self, other: Union[Portfolio, CompositePortfolio]) -> CompositePortfolio:
        if isinstance(other, Portfolio):
            return CompositePortfolio(self, other)
        elif isinstance(other, CompositePortfolio):
            return other.__add__(self)


# TODO: Portfolios for 3D data
class EquallyWeightedPortfolio(Portfolio):
    def __init__(self,
                 initial_cash: Union[int, float],
                 transaction_cost: Union[int, float],
                 trading_periods: int,
                 num_risky_assets: int):
        name: str = "ew"
        super().__init__(initial_cash,
                         transaction_cost,
                         trading_periods,
                         num_risky_assets,
                         name)
        self.action: ndarray = np.round(np.repeat(1 / (self.n - 1), self.n - 1), 6)

    # Todo: remove it and place it above
    def step(self, t: int,
             risky_asset_returns_t: ndarray,
             verbose: bool = False):
        self._verbose = verbose
        self.set_positions(self.action, t)

        # Indexing with `[t, 1:]` because we do not want to count two times the turnover
        self.turnover_per_risky_asset[t] = abs(self._weights[t, 1:] - self.positions_ante[t, 1:])
        self.turnover[t] = np.sum(self.turnover_per_risky_asset[t])
        self.trading_costs[t] = self.turnover[t] * self.tau * self.value_ante[t]
        self.trading_costs[t] = np.round(self.trading_costs[t], 6)

        self.unrolled_returns[t] = self.__compute_portfolio_returns(t, risky_asset_returns_t)
        reward_t: float = np.round(np.sum(self.unrolled_returns[t]), 5)
        self.rewards[t] = reward_t

        self.value[t] = np.round(self.value_ante[t] * (1 + reward_t), 5)
        self.cash_positions[t] = np.round(self.__compute_cash_positions(t), 5)
        self.__assert_portfolio_value(t)

        self.nav[t] = self.value[t] / self.initial_cash

        if t + 1 < self.trading_periods:
            self.value_ante[t + 1] = self.value[t]
            self.cash_positions_ante[t + 1] = self.cash_positions[t]
            self.nav_ante[t + 1] = self.nav_ante[t] * (1 + reward_t)
            assert np.allclose(self.nav[t], self.nav_ante[t + 1]), \
                f"NAVs don't match: {self.nav[t]} != {self.nav_ante[t + 1]}"

        if self._verbose:
            self.__print_progression()

        info: dict[str, float] = {'reward': reward_t,
                                  'nav': self.nav[t],
                                  'trading costs': self.trading_costs[t]}

        return reward_t, info

    def set_positions(self, action: Union[ndarray, list, tuple], t: int):
        """Checks if the sum of the risky asset weights are less or equal
        to 1, and raises an AssertionError message otherwise. The method
        also computes the weight of the riskless asset by calculating:
                                 w_{r_f, t} = 1 - α_t,
        where α_t = Σ_{i=1}^N w_{i, t}."""
        sum_risky_weights: float = np.round(np.sum(action), 5)
        assert sum_risky_weights <= 1, "Risky asset weights must sum up to 1 maximum."
        assert np.sum(action < 0) == 0, f"You cannot set negative weights ({np.sum(action < 0)})."
        rf_weight: float = np.round(1 - sum_risky_weights, 5)
        w_t: ndarray = np.append(rf_weight, action)
        self._weights[t] = w_t  # Can't use `@positions.setter` with slices
        if t + 1 < self.trading_periods:
            self.positions_ante[t + 1] = w_t

        if self._verbose:
            print("Positions:")
            print(self._weights)
            print("Ex-ante positions:")
            print(self.positions_ante)

    def __compute_portfolio_returns(self, t: int, risky_asset_returns_t: ndarray):
        self.__check_offset()
        # r_t: ndarray = np.append(0, risky_asset_returns_t)
        r_t: ndarray = risky_asset_returns_t
        w_t_minus_1: ndarray = self.positions_ante[t]
        turnover_per_position = np.append(0, self.turnover_per_risky_asset[t])  # To make shapes match
        return w_t_minus_1 * r_t - turnover_per_position * self.tau

        # Todo: unused, remove it

    def __compute_portfolio_return(self, t: int, risky_asset_returns_t: ndarray):
        self.__check_offset()
        # Todo: append the right risk-free rate
        # r_t: ndarray = np.append(0, risky_asset_returns_t)
        r_t: ndarray = risky_asset_returns_t
        w_t_minus_1: ndarray = self.positions_ante[t]
        return w_t_minus_1 @ r_t - self.turnover[t] * self.tau

    def __compute_cash_positions(self, t: int) -> ndarray:
        v_t: float = self.value[t]
        w_t: ndarray = self._weights[t]
        return v_t * w_t

    def __assert_portfolio_value(self, t: int):
        lhs: float = np.round(np.sum(self.cash_positions[t]), 5)
        rhs: float = self.value[t]
        cash_positions_dont_match: str = "np.sum(self.cash_positions[t]) !=" \
                                         " self.portfolio_value[t] since " \
                                         f"{lhs} != {rhs} at step {t}"
        assert np.allclose(lhs, rhs), cash_positions_dont_match

    def __check_offset(self):
        error_message: str = "The offset is 'None', please assign it an integer."
        assert self.offset is not None, error_message

    def __print_progression(self):
        print("Turnover (risky assets):")
        print(self.turnover_per_risky_asset)
        print("Turnover:")
        print(self.turnover)
        print("Trading costs:")
        print(self.trading_costs)
        print("Portfolio returns:")
        print(self.unrolled_returns)
        print("Rewards:")
        print(self.rewards)
        print("Portfolio's value:")
        print(self.value)
        print("Ex-ante portfolio's value:")
        print(self.value_ante)
        print("Cash positions:")
        print(self.cash_positions)
        print("Ex-ante cash positions:")
        print(self.cash_positions_ante)
        print("NAV:")
        print(self.nav)
        print("Ex-ante NAV:")
        print(self.nav_ante)


class TraderPortfolio(Portfolio):
    def __init__(self,
                 initial_cash: Union[int, float],
                 transaction_cost: Union[int, float],
                 trading_periods: int,
                 num_risky_assets: int):
        name: str = "trader"
        super().__init__(initial_cash,
                         transaction_cost,
                         trading_periods,
                         num_risky_assets,
                         name)

    def step(self,
             action: Union[ndarray, list, tuple],
             t: int,
             risky_asset_returns_t: ndarray,
             verbose: bool = False):
        self._verbose = verbose

        # Updates `positions_ante` and `_weights`
        self.set_positions(action, t)

        # Indexing with `[t, 1:]` because we do not want to count two times the turnover
        self.turnover_per_risky_asset[t] = abs(self._weights[t, 1:] - self.positions_ante[t, 1:])
        self.turnover[t] = np.sum(self.turnover_per_risky_asset[t])
        self.trading_costs[t] = self.turnover[t] * self.tau * self.value_ante[t]
        self.trading_costs[t] = np.round(self.trading_costs[t], 6)

        self.unrolled_returns[t] = self.__compute_portfolio_returns(t, risky_asset_returns_t)
        reward_t: float = np.round(np.sum(self.unrolled_returns[t]), 5)
        self.rewards[t] = reward_t

        self.value[t] = np.round(self.value_ante[t] * (1 + reward_t), 5)
        self.cash_positions[t] = np.round(self.__compute_cash_positions(t), 5)
        self.__assert_portfolio_value(t)

        self.nav[t] = self.value[t] / self.initial_cash

        if t + 1 < self.trading_periods:
            self.value_ante[t + 1] = self.value[t]
            self.cash_positions_ante[t + 1] = self.cash_positions[t]
            self.nav_ante[t + 1] = self.nav_ante[t] * (1 + reward_t)
            assert np.allclose(self.nav[t], self.nav_ante[t + 1]), \
                f"NAVs don't match: {self.nav[t]} != {self.nav_ante[t + 1]}"

        if self._verbose:
            self.__print_progression()

        # TODO: pass more info?
        info: dict[str, float] = {'reward': reward_t,
                                  'nav': self.nav[t],
                                  'trading costs': self.trading_costs[t]}

        return reward_t, info

    def set_positions(self, action: Union[ndarray, list, tuple], t: int):
        """Checks if the sum of the risky asset weights are less or equal
        to 1, and raises an AssertionError message otherwise. The method
        also computes the weight of the riskless asset by calculating:
                                 w_{r_f, t} = 1 - α_t,
        where α_t = Σ_{i=1}^N w_{i, t}."""
        sum_risky_weights: float = np.round(np.sum(action), 5)
        assert sum_risky_weights <= 1, "Risky asset weights must sum up to 1 maximum."
        rf_weight: float = np.round(1 - sum_risky_weights, 5)
        w_t: ndarray = np.append(rf_weight, action)
        self._weights[t] = w_t  # Can't use `@positions.setter` with slices
        if t + 1 < self.trading_periods:
            self.positions_ante[t + 1] = w_t

        if self._verbose:
            print("Positions:")
            print(self._weights)
            print("Ex-ante positions:")
            print(self.positions_ante)

    def __compute_portfolio_returns(self, t: int, risky_asset_returns_t: ndarray):
        self.__check_offset()
        # r_t: ndarray = np.append(0, risky_asset_returns_t)
        r_t: ndarray = risky_asset_returns_t
        w_t_minus_1: ndarray = self.positions_ante[t]
        turnover_per_position = np.append(0, self.turnover_per_risky_asset[t])  # To make shapes match
        return w_t_minus_1 * r_t - turnover_per_position * self.tau

    # Todo: unused, remove it
    def __compute_portfolio_return(self, t: int, risky_asset_returns_t: ndarray):
        self.__check_offset()
        # r_t: ndarray = np.append(0, risky_asset_returns_t)
        r_t: ndarray = risky_asset_returns_t
        w_t_minus_1: ndarray = self.positions_ante[t]
        return w_t_minus_1 @ r_t - self.turnover[t] * self.tau

    def __compute_cash_positions(self, t: int) -> ndarray:
        v_t: float = self.value[t]
        w_t: ndarray = self._weights[t]
        return v_t * w_t

    def __assert_portfolio_value(self, t: int):
        lhs: float = np.round(np.sum(self.cash_positions[t]), 5)
        rhs: float = self.value[t]
        cash_positions_dont_match: str = "np.sum(self.cash_positions[t]) !=" \
                                         " self.portfolio_value[t] since " \
                                         f"{lhs} != {rhs} at step {t}"
        assert np.allclose(lhs, rhs), cash_positions_dont_match

    def __check_offset(self):
        error_message: str = "The offset is 'None', please assign it an integer."
        assert self.offset is not None, error_message

    def __print_progression(self):
        print("Turnover (risky assets):")
        print(self.turnover_per_risky_asset)
        print("Turnover:")
        print(self.turnover)
        print("Trading costs:")
        print(self.trading_costs)
        print("Portfolio returns:")
        print(self.unrolled_returns)
        print("Rewards:")
        print(self.rewards)
        print("Portfolio's value:")
        print(self.value)
        print("Ex-ante portfolio's value:")
        print(self.value_ante)
        print("Cash positions:")
        print(self.cash_positions)
        print("Ex-ante cash positions:")
        print(self.cash_positions_ante)
        print("NAV:")
        print(self.nav)
        print("Ex-ante NAV:")
        print(self.nav_ante)


class CompositePortfolio:
    def __init__(self, *args: Optional[Union[Portfolio, CompositePortfolio]]):
        self.portfolios: list[Portfolio] = []
        for arg in args:
            check_type(arg, [Portfolio, CompositePortfolio, type(None)])
            if isinstance(arg, Portfolio):
                self.portfolios.append(arg)
            elif isinstance(arg, CompositePortfolio):
                li: list[Portfolio] = arg.portfolios
                self.portfolios += li

    def __add__(self, other: Union[Portfolio, CompositePortfolio]) -> CompositePortfolio:
        if isinstance(other, Portfolio):
            return CompositePortfolio(*self.portfolios, other)
        elif isinstance(other, CompositePortfolio):
            return CompositePortfolio(*self.portfolios, *other.portfolios)

    def __str__(self):
        n: int = len(self.portfolios)
        return f"<{__name__}.{self.__class__.__name__} object with {n} Portfolios at {hex(id(self))}>"

    __repr__ = __str__


class Simulator(ABC):
    """Interface for the TradingSimulator class."""
    @abstractmethod
    def reset(self, offset: int):
        pass

    @abstractmethod
    def step(self, action: Union[ndarray, list, tuple], verbose: bool = False):
        pass


class TradingSimulator(Simulator):
    def __init__(self,
                 initial_cash: Union[int, float],
                 transaction_cost: Union[int, float],
                 trading_periods: int,
                 risky_asset_returns: DataFrame,
                 risk_free_rate: DataFrame,
                 offset: Optional[int] = None,
                 baseline_portfolios: Optional[list[Portfolio]] = None):
        # Assignations
        self.initial_cash: Union[int, float] = initial_cash
        self.tau: Union[int, float] = transaction_cost
        self.trading_periods: int = trading_periods
        self.n: int = risky_asset_returns.shape[1]
        self.offset: int = offset
        self.risky_asset_returns: DataFrame = risky_asset_returns
        self.rf: DataFrame = risk_free_rate

        self.current_step: int = 0

        # Portfolio monitoring
        self.baseline_portfolios: Optional[list[Portfolio]] = baseline_portfolios
        self.num_baseline_portfolios: int = 0 if self.baseline_portfolios is None else len(self.baseline_portfolios)
        self.trader_portfolio: TraderPortfolio = TraderPortfolio(initial_cash=self.initial_cash,
                                                                 transaction_cost=self.tau,
                                                                 trading_periods=self.trading_periods,
                                                                 num_risky_assets=self.n)

        self.num_portfolios: int = 1 + self.num_baseline_portfolios

    def reset(self, offset: int):
        self.current_step = 0
        self.offset = offset
        self.trader_portfolio.reset(self.offset)
        if self.baseline_portfolios is not None:
            for ptf in self.baseline_portfolios:
                ptf.reset(self.offset)

    def step(self, action: Union[ndarray, list, tuple], verbose: bool = False):
        t: int = self.current_step
        r_t: ndarray = self.__get_risky_asset_returns(t)
        # Todo: self.offset + t or self.offset + t - 1?
        r_t = np.append(self.rf.iloc[self.offset + t - 1].values, r_t)
        rewards_t: dict[str, float] = {}

        if verbose:
            print('\n' + '-' * 25)
            print(f"Current step: {t}")

        reward_t, info = self.trader_portfolio.step(action, t, r_t, verbose)
        rewards_t[self.trader_portfolio.name] = reward_t

        for ptf in self.baseline_portfolios:
            rewards_t[ptf.name] = ptf.step(t, r_t, verbose)

        self.current_step += 1
        # If the NAV drops to 0, the episode ends with a loss:
        done: bool = (t > self.trading_periods) or (self.trader_portfolio.nav[t] < 0)
        # return reward_t, info, done
        return rewards_t, info, done

    def __get_risky_asset_returns(self, t: int) -> ndarray:
        if t == 0:
            return np.zeros((self.n,))
        else:  # Todo: Why offset + t - 1?
            return self.risky_asset_returns.iloc[self.offset + t - 1].to_numpy()


class TradingEnv(gym.Env):
    """
    :ivar action_space: The action space. It is defined as
    the portfolio allocation decision taken by the agent.
    The allocation is a weights' array applied to the risky
    assets. The rest is meant to go in the risk-free asset.

    :ivar observation_space: The observation space contains
    the minimal and maximal values of the features of each
    risky asset.
    Therefore, the observation space has shape (N, T=1, D)
    where,
        -> N is the number of risky assets,
        -> T=1 since it is an observation at time step t,
        -> D is the number of features.
    """
    metadata = {'render.modes': ['human']}
    continuous: bool = True

    # TODO: there should be a way to add benchmark portfolios, but the trader portfolio should be by default
    def __init__(self,
                 initial_cash: Union[int, float],
                 transaction_cost: Union[int, float],
                 dataset: PricesProcessor,
                 baseline_portfolios: Optional[list[Portfolio]] = None,
                 trading_periods: int = 252,
                 min_action: Union[int, float] = 0,
                 verbose: bool = False):
        super(TradingEnv, self).__init__()

        # Raise errors
        check_type(initial_cash, [int, float])
        check_type(transaction_cost, [int, float])
        check_type(dataset, [DataProcessor])
        check_type(trading_periods, [int])
        check_type(min_action, [int, float])
        if min_action > 0:
            raise ValueError(f"You must pass a negative value. You passed {min_action!r} instead.")
        check_type(verbose, [bool])

        # Attributes assignation
        self.initial_cash: Union[int, float] = initial_cash
        self.tau: Union[int, float] = transaction_cost
        self.trading_periods: int = trading_periods
        self.verbose: bool = verbose
        self.dataset: PricesProcessor = dataset
        self.dataset.trading_periods = self.trading_periods

        # Simulator instantiation
        # TODO: the simulator interacts with the dataset in a request-type fashion
        self.simulator: TradingSimulator = TradingSimulator(initial_cash=self.initial_cash,
                                                            transaction_cost=self.tau,
                                                            risky_asset_returns=dataset.returns1p,
                                                            risk_free_rate=dataset.rf,
                                                            baseline_portfolios=baseline_portfolios,
                                                            trading_periods=self.trading_periods)

        # Action space
        # N x 1 x 1 -> (N,)
        self.action_space = spaces.Box(low=min_action, high=1, shape=(self.dataset.num_assets,))

        # Observation space
        # TODO: min across time
        # N x 1 x D -> (N x D)
        min_vals: ndarray = np.nanmin(self.dataset.X, axis=1)
        max_vals: ndarray = np.nanmax(self.dataset.X, axis=1)
        self.observation_space = spaces.Box(min_vals, max_vals)

        self.offset: Optional[int] = None
        self.reset()

    def reset(self):
        self.offset = self.__generate_offset()
        self.dataset.reset(self.offset)
        self.simulator.reset(self.offset)
        return self.dataset.step()[0]

    def step(self, action: Union[ndarray, list, tuple]) -> tuple:
        assert self.action_space.contains(action), f"Invalid {action}."
        observation, end_episode = self.dataset.step()
        reward_for_each_portfolio, info, zero_nav = self.simulator.step(action, verbose=self.verbose)
        done: bool = end_episode or zero_nav
        return observation, reward_for_each_portfolio, done, info

    # TODO: define the render method
    def render(self, mode='human'):
        pass

    def __generate_offset(self) -> int:
        high = self.dataset.num_periods - self.trading_periods
        return np.random.randint(low=0, high=high)


if __name__ == '__main__':
    INITIAL_CASH: int = 1_000
    TRADING_PERIODS: int = 5  # Episode length
    TRANSACTION_COST: float = 1e-3

    ptf1 = TraderPortfolio(INITIAL_CASH, TRANSACTION_COST, TRADING_PERIODS, 3)
    ptf2 = EquallyWeightedPortfolio(INITIAL_CASH, TRANSACTION_COST, TRADING_PERIODS, 3)

    ptfs = ptf1 + ptf2
    print(ptfs.portfolios)

    # TODO: implement __radd__
    ptfs2 = ptfs + ptf2
    print(ptfs.portfolios)
    print(ptfs2.portfolios)

    ptfs2 = ptf2 + ptfs
    print(ptfs.portfolios)
    print(ptfs2.portfolios)

    ptfs3 = ptfs + ptfs2
    print(ptfs3.portfolios)