from __future__ import annotations

import gym
from gym import spaces
import numpy as np
from numpy.core import ndarray
import pandas as pd
from pandas.core.frame import DataFrame, Series
from typing import Optional, Annotated, Union
from src.envs.utils import check_type
from abc import ABC, abstractmethod


class DataProcessor(ABC):
    def __init__(self, data: DataFrame):
        check_type(data, [DataFrame])

    @abstractmethod
    def _process_data(self, *args):
        pass

    @staticmethod
    def to_period_index(data: DataFrame, freq: str):
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.to_period(freq)
        return data


class RiskFreeRateProcessor(DataProcessor):
    def __init__(self, data: DataFrame, freq: str = 'B'):
        super().__init__(data)
        self.rf: DataFrame = self._process_data(data, freq)

    def _process_data(self, data: DataFrame, freq: str):
        return self.__resample(data, freq)

    def __resample(self, data: DataFrame, freq: str) -> DataFrame:
        data = data.resample(freq).interpolate()
        data = self.to_period_index(data, freq)
        return data


class PricesProcessor(DataProcessor):
    def __init__(self, data: DataFrame, freq: str = 'B'):
        super().__init__(data)
        self.prices: DataFrame = self._process_data(data, freq)
        assert self.prices.isnull().sum().sum() == 0, "There are null values in the dataset."
        self.num_periods: int = self.prices.shape[0]
        self.num_risky_assets: int = self.prices.shape[1]
        # self.includes_risk_free_rate: bool = False
        # self.rf: Optional[DataFrame] = None

        # One-period returns
        self.returns1p: DataFrame = self.prices.pct_change().iloc[1:, :]

    def _process_data(self, data: DataFrame, freq: str):
        data = self.__resample(data, freq)
        data = self.__remove_columns_with_null_values(data)
        return data
        # self.__compute_technical_indicators()

    def __compute_returns(self):
        self.returns1p = self.prices.pct_change().iloc[1:, :]

    def __resample(self, data: DataFrame, freq: str) -> DataFrame:
        data = data.resample(freq).interpolate()
        data = self.to_period_index(data, freq)
        return data

    @staticmethod
    def __remove_columns_with_null_values(data: DataFrame) -> DataFrame:
        null_vals_per_col: Series = data.isnull().sum()
        total_num_null_vals: int = null_vals_per_col.sum()
        num_cols_with_null_vals: int = len(null_vals_per_col[null_vals_per_col > 0])

        if total_num_null_vals != 0:
            print(f"There are {total_num_null_vals} null values... "
                  f"Removing {num_cols_with_null_vals} columns.")
            columns_without_nulls: list[str] = list(null_vals_per_col[null_vals_per_col == 0].index)
            data = data[columns_without_nulls]

        return data

    # TODO: add `elif isinstance(other, RiskFreeRateProcessor):`
    def __add__(self, other) -> None:
        if isinstance(other, RiskFreeRateProcessor):
            if self.returns1p.index.equals(other.rf.index):
                if not other.rf.columns.isin(self.returns1p.columns)[0]:
                    return Dataset(self, other.rf)
                    # self.returns1p = pd.concat([other.rf, self.returns1p], axis=1)
                    # self.includes_risk_free_rate = True
                else:
                    print("Column has not been added to 'return1p' because it already exists.")
            else:
                freq: str = self.returns1p.freqstr
                rf: DataFrame = other.rf.resample(freq).interpolate()
                if rf.index.equals(self.returns1p.index):
                    self.__add__(rf)
                else:
                    raise Exception("Index must match!")


class Dataset:
    def __init__(self,
                 processed_assets: PricesProcessor,
                 risk_free_rate: Union[int, float, Series, DataFrame, RiskFreeRateProcessor] = 0):

        check_type(processed_assets, [PricesProcessor])
        check_type(risk_free_rate, [int, float, Series, DataFrame, RiskFreeRateProcessor])

        self.prices: DataFrame = processed_assets.prices
        self.returns1p: DataFrame = processed_assets.returns1p
        self.num_risky_assets: int = processed_assets.num_risky_assets
        self.num_periods: int = processed_assets.num_periods
        self.rf: DataFrame = self.__set_risk_free_rate(risk_free_rate)

    # TODO: truncate if risk-free asset is too long but periods correspond
    def __set_risk_free_rate(self, rf: Union[int, float, Series, DataFrame, RiskFreeRateProcessor]) -> DataFrame:
        is_constant_rate: bool = (isinstance(rf, float) or isinstance(rf, int))
        if is_constant_rate:
            return self.__propagate(rf)
        else:
            if isinstance(rf, RiskFreeRateProcessor):
                assert self.returns1p.index.equals(rf.rf.index), \
                    "The asset returns' Index does not match the risk-free rate Index."
                return rf.rf
            elif isinstance(rf, Series) or isinstance(rf, DataFrame):
                assert self.returns1p.index.equals(rf.index), \
                    "The asset returns' Index does not match the risk-free rate Index."
                if isinstance(rf, Series):
                    return rf.to_frame(name='rf')
                else:
                    return rf

    def __propagate(self, value: Union[int, float]):
        return pd.DataFrame(np.repeat(value, self.returns1p),
                            index=self.returns1p.index[1:],
                            columns=['rf'])


# class Portfolio:
#     """
#     A Portfolio is a collection of risky assets with
#     a risk-free asset that has metrics.
#
#     :param trading_periods: Trading periods of a trading
#     session. In other words, the episode length.
#     :param transaction_cost: Transaction costs. Assumed
#     to be constant over time.
#     :param rf: Risk-free rate.
#     """
#     def __init__(self,
#                  initial_cash: Union[int, float],
#                  risky_asset_prices: DataFrame,  # TODO: pass a data object
#                  trading_periods: int,
#                  transaction_cost: Union[int, float],
#                  rf: Union[int, float, Series, DataFrame] = 0,
#                  name: Optional[str] = None):
#         # Raise TypeError messages for debugging
#         check_type(initial_cash, [int, float])
#         check_type(risky_asset_prices, [DataFrame])
#         check_type(trading_periods, [int])
#         check_type(transaction_cost, [int, float])
#         check_type(rf, [int, float, Series, DataFrame])
#         check_type(name, [str, type(None)])
#
#         # Constant attributes
#         self.initial_cash: Union[int, float] = initial_cash
#         self.trading_periods: int = trading_periods
#         self.tau: Union[int, float] = transaction_cost
#         self.__id: str = hex(id(self))
#         self.name: str = self.__id if name is None else name
#
#         # TODO: Whilst creating a DataProcessor object, this could be removed
#         # Data
#         self.risky_asset_prices: DataFrame = risky_asset_prices
#         self.n_risky_assets: int = self.risky_asset_prices.shape[1]
#         # Risky assets plus a risk-free asset:
#         self.n_assets: int = self.n_risky_assets + 1
#         # The dataset height
#         self.whole_dataset_periods: int = len(self.risky_asset_prices)
#
#         self.offset: Optional[int] = None
#
#         # TODO: test if it works with a dynamic series
#         # Assign the risk-free rate correctly:
#         self.rf: DataFrame = self.__check_risk_free_rate(rf)
#
#         # TODO: Whilst creating a DataProcessor object, this could be removed
#         # Compute risk-asset dataset returns and concatenate them with the risk-free asset
#         self.risky_asset_returns: DataFrame = self.risky_asset_prices.pct_change().dropna()
#         self.all_asset_returns: DataFrame = pd.concat([self.rf, self.risky_asset_returns], axis=1)
#
#         # Initial weights are following allocations:
#         #   -> 100% in the bank account (risk-free asset), and
#         #   -> 0% in every other asset.
#         self.initial_weights: ndarray = np.append(1, np.repeat(0, self.n_risky_assets))
#         # The weights are the "actions" taken by the agent, or also the "positions" at time t:
#         self.__weights: ndarray = np.zeros((self.trading_periods, self.n_assets))
#         # `weights_ante` are the weights just before the period rebalancing, i.e., at t-1:
#         self.positions_ante: ndarray = np.zeros_like(self.__weights)
#         # Initially, all money is in the bank account. Thus, the risk-free asset:
#         self.positions_ante[0][0] = 1
#
#         # The weighted-returns of each asset in which the agent is long.
#         self.portfolio_returns: ndarray = np.zeros_like(self.__weights)
#
#         # The monetary value of each position the agent is in:
#         self.cash_positions_ante: ndarray = np.zeros_like(self.positions_ante)
#         self.cash_positions_ante[0][0] = self.initial_cash
#         self.cash_positions: ndarray = np.zeros_like(self.__weights)
#
#         # The monetary value of the portfolio, i.e., the sum of the cash positions at time t
#         self.portfolio_value_ante: ndarray = np.sum(self.cash_positions_ante, axis=1)
#         self.portfolio_value: ndarray = np.zeros_like(self.portfolio_value_ante)
#
#         # Rewards are used to compute the total return, thus they are the portfolio returns
#         self.rewards: ndarray = np.zeros(self.trading_periods)  # = Portfolio returns
#         self.nav: ndarray = np.ones(self.trading_periods)  # = Portfolio value of $1
#         self.nav_ante: ndarray = np.ones(self.trading_periods)
#         self.turnover_per_risky_asset: ndarray = np.zeros((self.trading_periods, self.n_risky_assets))
#         self.turnover: ndarray = np.zeros(self.trading_periods)
#         self.trading_costs: ndarray = np.zeros(self.trading_periods)  # Monetary value
#
#         self.__verbose: bool = False
#
#     def reset(self):
#         self.__weights.fill(0)
#         self.positions_ante.fill(0)
#         self.positions_ante[0][0] = 1
#
#         self.portfolio_returns.fill(0)
#
#         self.cash_positions_ante.fill(0)
#         self.cash_positions_ante[0][0] = self.initial_cash
#         self.cash_positions.fill(0)
#
#         # The portfolio value is the sum of the cash positions in each asset
#         self.portfolio_value_ante = np.sum(self.cash_positions_ante, axis=1)
#         self.portfolio_value.fill(0)
#
#         self.rewards.fill(0)
#         self.nav.fill(1)
#         self.nav_ante.fill(1)
#         self.turnover.fill(0)
#         self.turnover_per_risky_asset.fill(0)
#         self.trading_costs.fill(0)
#
#         self.__verbose = False
#
#     def step(self, action: Union[ndarray, list, tuple], t: int, verbose: bool = False) -> tuple[float, dict]:
#         self.__verbose = verbose
#
#         # Updates `positions_ante` and `__weights`
#         self.set_positions(action, t)
#
#         # Indexing with `[t, 1:]` because we do not want to count two times the turnover
#         self.turnover_per_risky_asset[t] = abs(self.__weights[t, 1:] - self.positions_ante[t, 1:])
#         self.turnover[t] = np.sum(self.turnover_per_risky_asset[t])
#         self.trading_costs[t] = self.turnover[t] * self.tau * self.portfolio_value_ante[t]
#         self.trading_costs[t] = np.round(self.trading_costs[t], 6)
#
#         self.portfolio_returns[t] = self.__compute_portfolio_returns(t)
#         reward_t: float = np.round(np.sum(self.portfolio_returns[t]), 5)
#         self.rewards[t] = reward_t
#
#         self.portfolio_value[t] = np.round(self.portfolio_value_ante[t] * (1 + reward_t), 5)
#         self.cash_positions[t] = np.round(self.__compute_cash_positions(t), 5)
#         self.__assert_portfolio_value(t)
#
#         self.nav[t] = self.portfolio_value[t] / self.initial_cash
#
#         if t + 1 < self.trading_periods:
#             self.portfolio_value_ante[t+1] = self.portfolio_value[t]
#             self.cash_positions_ante[t+1] = self.cash_positions[t]
#             self.nav_ante[t+1] = self.nav_ante[t] * (1 + reward_t)
#             assert np.allclose(self.nav[t], self.nav_ante[t+1]), \
#                 f"NAVs don't match: {self.nav[t]} != {self.nav_ante[t+1]}"
#
#         if self.__verbose:
#             self.__print_progression()
#
#         # TODO: pass more info?
#         info: dict[str, float] = {'reward': reward_t,
#                                   'nav': self.nav[t],
#                                   'trading costs': self.trading_costs[t]}
#
#         return reward_t, info
#
#     def __assert_portfolio_value(self, t: int):
#         lhs: float = np.round(np.sum(self.cash_positions[t]), 5)
#         rhs: float = self.portfolio_value[t]
#         cash_positions_dont_match: str = "np.sum(self.cash_positions[t]) !=" \
#                                          " self.portfolio_value[t] since " \
#                                          f"{lhs} != {rhs} at step {t}"
#         assert np.allclose(lhs, rhs), cash_positions_dont_match
#
#     def __print_progression(self):
#         print("Turnover (risky assets):")
#         print(self.turnover_per_risky_asset)
#         print("Turnover:")
#         print(self.turnover)
#         print("Trading costs:")
#         print(self.trading_costs)
#         print("Portfolio returns:")
#         print(self.portfolio_returns)
#         print("Rewards:")
#         print(self.rewards)
#         print("Portfolio's value:")
#         print(self.portfolio_value)
#         print("Ex-ante portfolio's value:")
#         print(self.portfolio_value_ante)
#         print("Cash positions:")
#         print(self.cash_positions)
#         print("Ex-ante cash positions:")
#         print(self.cash_positions_ante)
#         print("NAV:")
#         print(self.nav)
#         print("Ex-ante NAV:")
#         print(self.nav_ante)
#
#     def __compute_portfolio_return(self, t: int):
#         self.__check_offset()
#         r_t: ndarray = self.__asset_returns(t)
#         w_t_minus_1: ndarray = self.positions_ante[t]
#         return w_t_minus_1 @ r_t - self.turnover[t] * self.tau
#
#     def __compute_portfolio_returns(self, t: int):
#         self.__check_offset()
#         r_t: ndarray = self.__asset_returns(t)
#         w_t_minus_1: ndarray = self.positions_ante[t]
#         turnover_per_position = np.append(0, self.turnover_per_risky_asset[t])  # To make shapes match
#         return w_t_minus_1 * r_t - turnover_per_position * self.tau
#
#     def __compute_cash_positions(self, t: int) -> ndarray:
#         v_t: float = self.portfolio_value[t]
#         w_t: ndarray = self.positions[t]
#         return v_t * w_t
#
#     def __check_risk_free_rate(self, rf: Union[int, float, Series, DataFrame]) -> DataFrame:
#         is_constant_rate: bool = (isinstance(rf, float) or isinstance(rf, int))
#         return self.__propagate(rf) if is_constant_rate else rf
#
#     def __propagate(self, value: Union[int, float]):
#         return pd.DataFrame(np.repeat(value, self.whole_dataset_periods - 1),
#                             index=self.risky_asset_prices.index[1:],
#                             columns=['rf'])
#
#     def __asset_returns(self, t: int) -> ndarray:
#         if t == 0:
#             return np.zeros((self.n_assets,))
#         else:
#             return self.all_asset_returns.iloc[self.offset + t - 1].to_numpy()
#
#     def __check_offset(self):
#         error_message: str = "The offset is 'None', please assign it an integer."
#         assert self.offset is not None, error_message
#
#     @property
#     def positions(self):
#         """The actions the agent takes is the portion
#         of money to allocate in each asset. Thus, the
#         weights or the positions."""
#         return self.__weights
#
#     def set_positions(self, action: Union[ndarray, list, tuple], t: Annotated[int, "time step"]):
#         """Checks if the sum of the risky asset weights are less or equal
#         to 1, and raises an AssertionError message otherwise. The method
#         also computes the weight of the riskless asset by calculating:
#                                  w_{r_f, t} = 1 - α_t,
#         where α_t = Σ_{i=1}^N w_{i, t}."""
#         sum_risky_weights: float = np.round(np.sum(action), 5)
#         assert sum_risky_weights <= 1, "Risky asset weights must sum up to 1 maximum."
#         rf_weight: float = np.round(1 - sum_risky_weights, 5)
#         w_t: ndarray = np.append(rf_weight, action)
#         self.__weights[t] = w_t  # Can't use `@positions.setter` with slices
#         if t + 1 < self.trading_periods:
#             self.positions_ante[t + 1] = w_t
#
#         if self.__verbose:
#             print("Positions:")
#             print(self.__weights)
#             print("Ex-ante positions:")
#             print(self.positions_ante)


class Portfolio:
    """
    A Portfolio is a collection of risky assets with
    a risk-free asset that has metrics.

    :param trading_periods: Trading periods of a trading
    session. In other words, the episode length.
    :param transaction_cost: Transaction costs. Assumed
    to be constant over time.
    :param rf: Risk-free rate.
    """
    def __init__(self,
                 initial_cash: Union[int, float],
                 dataset: DataProcessor,
                 trading_periods: int,
                 transaction_cost: Union[int, float],
                 rf: Union[int, float, Series, DataFrame] = 0,
                 name: Optional[str] = None):
        # Raise TypeError messages for debugging
        check_type(initial_cash, [int, float])
        check_type(dataset, [DataProcessor])
        check_type(trading_periods, [int])
        check_type(transaction_cost, [int, float])
        check_type(rf, [int, float, Series, DataFrame])
        check_type(name, [str, type(None)])

        # Constant attributes
        self.initial_cash: Union[int, float] = initial_cash
        self.trading_periods: int = trading_periods
        self.tau: Union[int, float] = transaction_cost
        self.__id: str = hex(id(self))
        self.name: str = self.__id if name is None else name

        # TODO: Whilst creating a DataProcessor object, this could be removed
        # Data
        self.dataset: DataProcessor = dataset
        # self.risky_asset_prices: DataFrame = risky_asset_prices
        self.n_risky_assets: int = self.risky_asset_prices.shape[1]
        # Risky assets plus a risk-free asset:
        self.n_assets: int = self.n_risky_assets + 1
        # The dataset height
        self.whole_dataset_periods: int = len(self.risky_asset_prices)

        self.offset: Optional[int] = None

        # TODO: test if it works with a dynamic series
        # Assign the risk-free rate correctly:
        self.rf: DataFrame = self.__check_risk_free_rate(rf)

        # TODO: Whilst creating a DataProcessor object, this could be removed
        # Compute risk-asset dataset returns and concatenate them with the risk-free asset
        self.risky_asset_returns: DataFrame = self.risky_asset_prices.pct_change().dropna()
        self.all_asset_returns: DataFrame = pd.concat([self.rf, self.risky_asset_returns], axis=1)

        # Initial weights are following allocations:
        #   -> 100% in the bank account (risk-free asset), and
        #   -> 0% in every other asset.
        self.initial_weights: ndarray = np.append(1, np.repeat(0, self.n_risky_assets))
        # The weights are the "actions" taken by the agent, or also the "positions" at time t:
        self.__weights: ndarray = np.zeros((self.trading_periods, self.n_assets))
        # `weights_ante` are the weights just before the period rebalancing, i.e., at t-1:
        self.positions_ante: ndarray = np.zeros_like(self.__weights)
        # Initially, all money is in the bank account. Thus, the risk-free asset:
        self.positions_ante[0][0] = 1

        # The weighted-returns of each asset in which the agent is long.
        self.portfolio_returns: ndarray = np.zeros_like(self.__weights)

        # The monetary value of each position the agent is in:
        self.cash_positions_ante: ndarray = np.zeros_like(self.positions_ante)
        self.cash_positions_ante[0][0] = self.initial_cash
        self.cash_positions: ndarray = np.zeros_like(self.__weights)

        # The monetary value of the portfolio, i.e., the sum of the cash positions at time t
        self.portfolio_value_ante: ndarray = np.sum(self.cash_positions_ante, axis=1)
        self.portfolio_value: ndarray = np.zeros_like(self.portfolio_value_ante)

        # Rewards are used to compute the total return, thus they are the portfolio returns
        self.rewards: ndarray = np.zeros(self.trading_periods)  # = Portfolio returns
        self.nav: ndarray = np.ones(self.trading_periods)  # = Portfolio value of $1
        self.nav_ante: ndarray = np.ones(self.trading_periods)
        self.turnover_per_risky_asset: ndarray = np.zeros((self.trading_periods, self.n_risky_assets))
        self.turnover: ndarray = np.zeros(self.trading_periods)
        self.trading_costs: ndarray = np.zeros(self.trading_periods)  # Monetary value

        self.__verbose: bool = False

    def reset(self):
        self.__weights.fill(0)
        self.positions_ante.fill(0)
        self.positions_ante[0][0] = 1

        self.portfolio_returns.fill(0)

        self.cash_positions_ante.fill(0)
        self.cash_positions_ante[0][0] = self.initial_cash
        self.cash_positions.fill(0)

        # The portfolio value is the sum of the cash positions in each asset
        self.portfolio_value_ante = np.sum(self.cash_positions_ante, axis=1)
        self.portfolio_value.fill(0)

        self.rewards.fill(0)
        self.nav.fill(1)
        self.nav_ante.fill(1)
        self.turnover.fill(0)
        self.turnover_per_risky_asset.fill(0)
        self.trading_costs.fill(0)

        self.__verbose = False

    def step(self, action: Union[ndarray, list, tuple], t: int, verbose: bool = False) -> tuple[float, dict]:
        self.__verbose = verbose

        # Updates `positions_ante` and `__weights`
        self.set_positions(action, t)

        # Indexing with `[t, 1:]` because we do not want to count two times the turnover
        self.turnover_per_risky_asset[t] = abs(self.__weights[t, 1:] - self.positions_ante[t, 1:])
        self.turnover[t] = np.sum(self.turnover_per_risky_asset[t])
        self.trading_costs[t] = self.turnover[t] * self.tau * self.portfolio_value_ante[t]
        self.trading_costs[t] = np.round(self.trading_costs[t], 6)

        self.portfolio_returns[t] = self.__compute_portfolio_returns(t)
        reward_t: float = np.round(np.sum(self.portfolio_returns[t]), 5)
        self.rewards[t] = reward_t

        self.portfolio_value[t] = np.round(self.portfolio_value_ante[t] * (1 + reward_t), 5)
        self.cash_positions[t] = np.round(self.__compute_cash_positions(t), 5)
        self.__assert_portfolio_value(t)

        self.nav[t] = self.portfolio_value[t] / self.initial_cash

        if t + 1 < self.trading_periods:
            self.portfolio_value_ante[t+1] = self.portfolio_value[t]
            self.cash_positions_ante[t+1] = self.cash_positions[t]
            self.nav_ante[t+1] = self.nav_ante[t] * (1 + reward_t)
            assert np.allclose(self.nav[t], self.nav_ante[t+1]), \
                f"NAVs don't match: {self.nav[t]} != {self.nav_ante[t+1]}"

        if self.__verbose:
            self.__print_progression()

        # TODO: pass more info?
        info: dict[str, float] = {'reward': reward_t,
                                  'nav': self.nav[t],
                                  'trading costs': self.trading_costs[t]}

        return reward_t, info

    def __assert_portfolio_value(self, t: int):
        lhs: float = np.round(np.sum(self.cash_positions[t]), 5)
        rhs: float = self.portfolio_value[t]
        cash_positions_dont_match: str = "np.sum(self.cash_positions[t]) !=" \
                                         " self.portfolio_value[t] since " \
                                         f"{lhs} != {rhs} at step {t}"
        assert np.allclose(lhs, rhs), cash_positions_dont_match

    def __print_progression(self):
        print("Turnover (risky assets):")
        print(self.turnover_per_risky_asset)
        print("Turnover:")
        print(self.turnover)
        print("Trading costs:")
        print(self.trading_costs)
        print("Portfolio returns:")
        print(self.portfolio_returns)
        print("Rewards:")
        print(self.rewards)
        print("Portfolio's value:")
        print(self.portfolio_value)
        print("Ex-ante portfolio's value:")
        print(self.portfolio_value_ante)
        print("Cash positions:")
        print(self.cash_positions)
        print("Ex-ante cash positions:")
        print(self.cash_positions_ante)
        print("NAV:")
        print(self.nav)
        print("Ex-ante NAV:")
        print(self.nav_ante)

    def __compute_portfolio_return(self, t: int):
        self.__check_offset()
        r_t: ndarray = self.__asset_returns(t)
        w_t_minus_1: ndarray = self.positions_ante[t]
        return w_t_minus_1 @ r_t - self.turnover[t] * self.tau

    def __compute_portfolio_returns(self, t: int):
        self.__check_offset()
        r_t: ndarray = self.__asset_returns(t)
        w_t_minus_1: ndarray = self.positions_ante[t]
        turnover_per_position = np.append(0, self.turnover_per_risky_asset[t])  # To make shapes match
        return w_t_minus_1 * r_t - turnover_per_position * self.tau

    def __compute_cash_positions(self, t: int) -> ndarray:
        v_t: float = self.portfolio_value[t]
        w_t: ndarray = self.positions[t]
        return v_t * w_t

    def __check_risk_free_rate(self, rf: Union[int, float, Series, DataFrame]) -> DataFrame:
        is_constant_rate: bool = (isinstance(rf, float) or isinstance(rf, int))
        return self.__propagate(rf) if is_constant_rate else rf

    def __propagate(self, value: Union[int, float]):
        return pd.DataFrame(np.repeat(value, self.whole_dataset_periods - 1),
                            index=self.risky_asset_prices.index[1:],
                            columns=['rf'])

    def __asset_returns(self, t: int) -> ndarray:
        if t == 0:
            return np.zeros((self.n_assets,))
        else:
            return self.all_asset_returns.iloc[self.offset + t - 1].to_numpy()

    def __check_offset(self):
        error_message: str = "The offset is 'None', please assign it an integer."
        assert self.offset is not None, error_message

    @property
    def positions(self):
        """The actions the agent takes is the portion
        of money to allocate in each asset. Thus, the
        weights or the positions."""
        return self.__weights

    def set_positions(self, action: Union[ndarray, list, tuple], t: Annotated[int, "time step"]):
        """Checks if the sum of the risky asset weights are less or equal
        to 1, and raises an AssertionError message otherwise. The method
        also computes the weight of the riskless asset by calculating:
                                 w_{r_f, t} = 1 - α_t,
        where α_t = Σ_{i=1}^N w_{i, t}."""
        sum_risky_weights: float = np.round(np.sum(action), 5)
        # TODO: allow more when short-selling permitted
        # assert sum_risky_weights <= 1 * (1 + (-1) * self.min_action), "Risky asset weights must sum up to 1 maximum."
        assert sum_risky_weights <= 1, "Risky asset weights must sum up to 1 maximum."
        rf_weight: float = np.round(1 - sum_risky_weights, 5)
        w_t: ndarray = np.append(rf_weight, action)
        self.__weights[t] = w_t  # Can't use `@positions.setter` with slices
        if t + 1 < self.trading_periods:
            self.positions_ante[t + 1] = w_t

        if self.__verbose:
            print("Positions:")
            print(self.__weights)
            print("Ex-ante positions:")
            print(self.positions_ante)


class TradingSimulator:
    """
    Simulates the trading session. Tracks
    Portfolio objects' evolution.
    """
    def __init__(self,
                 trader_portfolio: Portfolio,  # TODO: find a way to aggregate portfolios
                 baseline_portfolio: Optional[Portfolio] = None,
                 trading_periods: int = 252,
                 offset: int = -1):
        self.trading_periods: int = trading_periods
        self.current_step: int = 0
        self.trader_portfolio: Portfolio = trader_portfolio
        self.baseline_portfolio: Portfolio = baseline_portfolio
        self.offset: int = self.__generate_offset() if offset == -1 else offset
        self.trader_portfolio.offset = self.offset
        if self.baseline_portfolio is not None:
            self.baseline_portfolio.offset = self.offset

    def reset(self):
        self.current_step = 0

        # Update the offset
        self.offset = self.__generate_offset()

        # Reset portfolios
        self.trader_portfolio.reset()
        self.trader_portfolio.offset = self.offset
        if self.baseline_portfolio is not None:
            self.baseline_portfolio.reset()
            self.baseline_portfolio.offset = self.offset

    def step(self, action: Union[ndarray, list, tuple], verbose: bool = False) -> tuple[ndarray, float, dict, bool]:
        t: int = self.current_step

        if verbose:
            print('\n' + '-' * 25)
            print(f"Current step: {t}")

        reward_t, info = self.trader_portfolio.step(action, t, verbose)

        if verbose:
            print('\n'.join([f"{k}: {v}" for k, v in info.items()]))

        observation: ndarray = self.trader_portfolio.risky_asset_returns.iloc[self.offset + t].to_numpy()
        self.current_step += 1
        # If the NAV drops to 0, the episode ends with a loss:
        done: bool = (t > self.trading_periods) or (self.trader_portfolio.nav[t] < 0)
        return observation, reward_t, info, done

    def __generate_offset(self) -> int:
        high = self.trader_portfolio.whole_dataset_periods - self.trading_periods
        return np.random.randint(low=0, high=high)


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trader_portfolio: Portfolio,
                 baseline_portfolio: Optional[Portfolio] = None,
                 trading_periods: int = 252,
                 min_action: Union[int, float] = 0,
                 verbose: bool = False):
        check_type(trader_portfolio, [Portfolio])
        check_type(trading_periods, [int])
        check_type(min_action, [int, float])
        check_type(verbose, [bool])

        self.trading_periods: int = trading_periods
        self.verbose: bool = verbose
        self.simulator: TradingSimulator = TradingSimulator(trader_portfolio,
                                                            baseline_portfolio=baseline_portfolio,
                                                            trading_periods=self.trading_periods)
        if min_action > 0:
            raise ValueError(f"You must pass a negative value. You passed {min_action!r} instead.")

        # Action space
        self.__min_action: Union[int, float] = min_action
        n_risky_assets: int = self.simulator.trader_portfolio.n_risky_assets
        self.action_space = spaces.Box(low=self.__min_action, high=1, shape=(n_risky_assets,))

        # TODO: define the observation space
        # Observation space
        data_min_values: ndarray = trader_portfolio.risky_asset_returns.min().to_numpy()
        data_max_values: ndarray = trader_portfolio.risky_asset_returns.max().to_numpy()
        self.observation_space = spaces.Box(data_min_values, data_max_values)

        self.reset()

    def reset(self):
        self.simulator.reset()

    def step(self, action: Union[ndarray, list, tuple]) -> tuple:
        assert self.action_space.contains(action), f"Invalid {action}."
        observation, reward, info, done = self.simulator.step(action, verbose=self.verbose)
        return observation, reward, done, info

    # TODO: define the render method
    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    # Constants
    INITIAL_CASH = 1_000
    TRADING_PERIODS: int = 5  # Episode length
    TRANSACTION_COST: float = 1e-3

    print("\nImport data:")
    data_ = pd.read_csv('../../data/sp500_closefull.csv', parse_dates=['Date'], index_col=['Date'])
    data_.info()

    print("\nTesting the Portfolio object:")
    ptf_data: DataFrame = data_[['AAPL', 'MSFT']].round(2)
    ptf = Portfolio(INITIAL_CASH, ptf_data, TRADING_PERIODS, TRANSACTION_COST)
    print(f"Portfolio's name: {ptf.name!r}")
    print(ptf.all_asset_returns.tail())

    # def get_rf_weight(array: ndarray):
    #     return np.append((1 - array.sum(axis=1)).reshape(-1, 1), array, axis=1)
    #
    # positions: ndarray = np.array([[0.3, 0.7],
    #                                [0.3, 0.6],
    #                                [0.2, 0.5]])
    #
    # positions = get_rf_weight(positions)
    #
    # positions_ante: ndarray = np.array([[0., 0.],
    #                                     [0.3, 0.7],
    #                                     [0.3, 0.6]])
    #
    # positions_ante = get_rf_weight(positions_ante)
    # risky_returns = ptf.all_asset_returns.iloc[0:2, :].to_numpy()
    #
    # # Portfolio cash value
    # v0_start = INITIAL_CASH
    # # Portfolio cash value positions
    # c0_start = positions_ante[0] * v0_start
    # # Rebalancing at the end of the period
    # w_init = positions_ante[0]
    # w_0 = positions[0]
    # turnover0 = np.append(0, abs(w_0[1:] - w_init[1:]))
    # r_0 = np.repeat(0., 3)
    # c0_end = (w_0 * (1 + r_0) - turnover0 * TRANSACTION_COST) * v0_start
    # v0_end = np.sum(c0_end)
    #
    # v1_start = v0_end
    # w_1 = positions[1]
    # r_1 = risky_returns[0]
    # turnover1 = np.append(0, abs(w_1[1:] - w_0[1:]))
    # c1_end = v1_start * (w_0 * (1 + r_1) - turnover1 * TRANSACTION_COST)
    # v1_end = np.sum(c1_end)
    #
    # v2_start = v1_end
    # w_2 = positions[2]
    # r_2 = risky_returns[1]
    # turnover2 = np.append(0, abs(w_2[1:] - w_1[1:]))
    # c2_end = v1_start * (w_1 * (1 + r_2) - turnover2 * TRANSACTION_COST)
    # v2_end = np.sum(c2_end)

    print("\nTesting the TradingSimulator object:")
    sim = TradingSimulator(ptf, trading_periods=TRADING_PERIODS, offset=1)
    print("Is the portfolio passed as an argument the same as the "
          "one assigned to the attribute? "
          f"{ptf.name == sim.trader_portfolio.name}.")
    # sim.step([0.3, 0.7], verbose=True)
    # # AssertionError: Weights must sum up to 1 maximum.
    # # sim.step([0.8, 0.3])
    # sim.step((0.3, 0.6), verbose=True)
    # sim.step((0.2, 0.5), verbose=True)

    n_risk_assets: int = ptf_data.shape[1]
    actions = np.random.rand(TRADING_PERIODS, n_risk_assets)
    factor = 1 + np.random.randint(low=0, high=101, size=TRADING_PERIODS) / 100
    normalized_actions = np.round(actions / (np.sum(actions, axis=1) * factor).reshape(-1, 1), 2)

    for i in range(TRADING_PERIODS):
        sim.step(normalized_actions[i], verbose=True)

    dat = PricesProcessor(data_)
    rf_df = pd.DataFrame(data=np.repeat(0, dat.returns1p.shape[0]), index=dat.returns1p.index, columns=['rf'])
    rf_ = RiskFreeRateProcessor(rf_df, 'B')
    dat + rf_  # rf_ OK
    dat + rf_  # Should not add column


