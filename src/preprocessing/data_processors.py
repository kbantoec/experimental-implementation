import pandas as pd
from pandas_datareader import DataReader
import talib as ta
import numpy as np
from pathlib import Path
import tensorflow as tf
from src.envs.utils import check_type

# Type hints
from numpy.core import ndarray
from pandas.core.frame import DataFrame
from pandas.core.resample import Period
from typing import Optional, Union

DIR_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = DIR_PATH / 'data'


class DataProcessor:
    def __init__(self,
                 data_path: Path,
                 filename: str,
                 key: Optional[str] = None,
                 rf: Optional[DataFrame] = None,  # TODO: allow propagation of int or float
                 macroindicators: Optional[DataFrame] = None,
                 normalize: bool = True,
                 freq: str = 'B',
                 trading_periods: Optional[int] = None):
        check_type(data_path, [Path])
        check_type(filename, [str])
        check_type(key, [str, type(None)])
        check_type(rf, [DataFrame, type(None)])
        check_type(macroindicators, [DataFrame, type(None)])
        check_type(normalize, [bool])
        check_type(freq, [str])
        check_type(trading_periods, [int, type(None)])

        self.macroindicators: Optional[DataFrame] = macroindicators
        self.rf: Optional[DataFrame] = rf
        self.freq: str = freq
        filepath: str = str(data_path / filename)
        self.data: DataFrame = self.load(filepath, key)
        self.normalize: bool = normalize
        self.current_step: int = 0
        self.offset: Optional[int] = None
        self.X: Optional[ndarray] = None
        self.trading_periods: Optional[int] = trading_periods

    def reset(self, offset: int):
        check_type(offset, [int])
        self.current_step = 0
        self.offset = offset

    def step(self):
        t: int = self.offset + self.current_step
        observation = self.X[:, t, :]
        done: bool = self.current_step > self.trading_periods
        self.current_step += 1
        return observation, done

    def _process(self, key: str, freq: str) -> DataFrame:
        assert key in self.__dict__.keys(), f"Attribute {key} does not exists."
        if self.__dict__[key] is not None:
            data = self.__dict__[key]
            data = data.resample(freq).interpolate()
            if isinstance(data.index, pd.DatetimeIndex):
                data.index = data.index.to_period(freq)
                print(f"Attribute {key!r} index has been converted to a 'pandas.PeriodIndex'.")
            num_nan: int = data.isnull().sum().sum()
            self._print_nan_message(num_nan, key)
            return data

    def _truncate(self):
        for key in ('rf', 'macroindicators'):
            attr = self.__dict__[key]
            if self.__dict__[key] is not None:
                if not self.data.index.equals(attr.index):
                    start: Period = max(attr.index.min(), self.data.index.min())
                    end: Period = min(attr.index.max(), self.data.index.max())
                    self.data = self.data.loc[start:end, :]
                    self.__dict__[key] = attr.loc[start:end, :]
                    assert self.data.index.equals(self.__dict__[key].index), \
                        f"Index of {key!r} attribute does not match the index of the data."

    @staticmethod
    def load(filepath: str, key: Optional[str] = None) -> Union[ndarray, DataFrame]:
        check_type(filepath, [str])
        if filepath.endswith('.npy'):
            data = np.load(filepath)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath, parse_dates=['Date'], index_col=['Date'])
        elif filepath.endswith('.h5'):
            store = pd.HDFStore(filepath, mode='r')
            data = store.get(key)
        else:
            data = None
            print('Not recognized extension.')
        return data

    @staticmethod
    def _print_nan_message(num_nan: int, key: str) -> None:
        warning_msg: str = f" No missing values left."

        if num_nan > 0:
            warning_msg = f" But there are {num_nan} missing values left."

        print(f"Data from {key!r} attribute has been processed.{warning_msg}")

    def __str__(self):
        has_rf: str = ''
        has_macro: str = ''
        if self.rf is not None:
            has_rf = "[1] risk-free rate "
        if self.macroindicators is not None:
            has_macro = f"[{self.macroindicators.shape[1]}] macro-indicators "
        has_attrs = has_rf + has_macro
        return f"<{__name__}.{self.__class__} {has_attrs}at {hex(id(self))}"


class PricesProcessor(DataProcessor):
    """
    :ivar X:
      -> X[:, 0] is the risk-free rate at time t
      -> X[:, 1] is the USARECDM recession binary indicator
      -> X[:, 2] is the 1-period return

    """
    def __init__(self,
                 data_path: Path,
                 filename: str,
                 key: Optional[str] = None,
                 rf=None,
                 macroindicators=None,
                 normalize: bool = True,
                 freq: str = 'B'):
        super().__init__(data_path, filename, key, rf, macroindicators, normalize, freq)

        self.data = self._process('data', self.freq)
        self.rf = self._process('rf', self.freq)
        self.macroindicators = self._process('macroindicators', self.freq)
        self._truncate()
        self.num_periods, self.num_assets = self.data.shape
        self.index = self.data.index
        self.tickers: list[str] = list(self.data.columns)
        self.num_features: Optional[int] = None
        self.returns1p: DataFrame = self.data.pct_change()

    def create_features(self) -> ndarray:
        multiperiod_returns: list[int] = [1, 2, 5, 10, 21]
        multiperiod_std: list[int] = [5, 10, 21]

        add_rf: int = 0 if self.rf is None else 1
        num_indicators: int = self.macroindicators.shape[1] if self.macroindicators is not None else 0

        N: int = self.num_assets
        T: int = self.num_periods
        D: int = len(multiperiod_std) + len(multiperiod_returns) + add_rf + num_indicators
        self.num_features = D

        X: ndarray = np.empty((N, T, D))
        for t, ticker in enumerate(self.data):
            df = self.data.loc[:, ticker].copy().to_frame(name='prices')
            if isinstance(self.rf, DataFrame):
                df['rf'] = self.rf.to_numpy()
            if isinstance(self.macroindicators, DataFrame):
                df = pd.concat([df, self.macroindicators], axis=1)
            else:
                print("Did not add the risk-free rate.")
            for lag in multiperiod_returns:
                df[f'returns{lag}p'] = df.prices.pct_change(lag)
            for lag in multiperiod_std:
                df[f'std{lag}p'] = df.prices.pct_change(lag)
            df.drop(['prices'], axis=1, inplace=True)
            X[t] = df.to_numpy()
        return X

    def save_tensor(self, out_filename: str):
        with open(DATA_PATH / f'{out_filename}.npy', 'wb') as f:
            np.save(f, self.X)

    def load_tensor(self, path: [Path, str]):
        self.X = np.load(path)


if __name__ == '__main__':
    if tf.test.is_gpu_available():
        print(tf.test.gpu_device_name())
        print(f"Cuda GPU: {tf.test.is_built_with_cuda()}")
    for device in tf.config.experimental.list_physical_devices('GPU'):
        print(f"{device.device_type}: {device.name}")

    # Importing data
    sp500_path: str = str(DATA_PATH / 'sp500_closefull.csv')
    data = pd.read_csv(DATA_PATH / 'sp500_closefull.csv', parse_dates=['Date'], index_col=['Date'])
    store = pd.HDFStore(DATA_PATH / 'DGS1MO.h5')
    annual_rf = pd.DataFrame(store.get('rf'))
    store.close()
    store = pd.HDFStore(DATA_PATH / 'macro.h5')
    indicators = pd.DataFrame(store.get('USARECDM'))
    store.close()
    prices = PricesProcessor(DATA_PATH,
                             'sp500_closefull.csv',
                             rf=annual_rf.loc[:, ['DGS1MO_annual']],
                             macroindicators=indicators)
    # prices.create_features()
    # prices.save_tensor('tensor')
    prices.load_tensor(DATA_PATH / 'tensor.npy')
    X = tf.Variable(np.copy(prices.X))
    print(f"X.shape -> {X.shape}")
    prices.trading_periods = 252
    prices.reset(345)
    obs, done = prices.step()
    print(f"t={prices.current_step-1}: obs.shape -> {obs.shape}, done: {done}, t={prices.current_step}")

    # # Resample data
    # data = data.resample('B').interpolate()
    # data.index = data.index.to_period('B')
    # # TODO: How to handle null values?
    # # `data.isnull().sum().sum()` -> 43746
    # T: int = len(data)
    # N: int = data.shape[1]
    # D: int = 6

    # # TODO: Build a DataProcessor.from_prices() .from_ohlc()
    # if (DATA_PATH / 'tensor.npy').exists():
    #     X = np.load(DATA_PATH / 'tensor.npy')
    # else:
    #     # TODO: add the risk-free asset and inflation rate
    #     X = np.empty((N, T, D))
    #     for t, ticker in enumerate(data):
    #         df = data.loc[:, ticker].to_frame(name='price')
    #         df['returns1p'] = df.price.pct_change()
    #         df['returns2p'] = df.price.pct_change(2)
    #         df['returns5p'] = df.price.pct_change(5)
    #         df['returns10p'] = df.price.pct_change(10)
    #         df['returns21p'] = df.price.pct_change(21)
    #         X[t] = df.to_numpy()
    #
    #     with open(DATA_PATH / 'tensor.npy', 'wb') as f:
    #         np.save(f, X)
    #
    # # TODO: How to store X as a tensorflow variable?
    # X = tf.Variable(X)

    # # rf = DataReader('DGS1MO', data_source='fred', start='2000')
    # # rf = rf.resample('B').interpolate()
    # # rf.index = rf.index.to_period('B')
    # # rf = rf.div(100)
    # # rf = rf.add_suffix('_annual')
    # # rf['DGS1MO_daily'] = rf.DGS1MO_annual.div(12 * 21)
    # store = pd.HDFStore(DATA_PATH / 'DGS1MO.h5')
    # if (DATA_PATH / 'DGS1MO.h5').exists():
    #     rf = store['rf']
    # else:
    #     rf = DataReader('DGS1MO', data_source='fred', start='2000')
    #     rf = rf.resample('B').interpolate()
    #     rf.index = rf.index.to_period('B')
    #     rf = rf.div(100)
    #     rf = rf.add_suffix('_annual')
    #     rf['DGS1MO_daily'] = rf.DGS1MO_annual.div(12 * 21)
    #     store['rf'] = rf
    # store.close()

    # # rf.to_csv(DATA_PATH / 'rf.csv')
    # # https://fred.stlouisfed.org/series/USARECDM
    # # A value of 1 is a recessionary period, while a value of 0 is an expansionary period.
    # macrostore = pd.HDFStore(DATA_PATH / 'macro.h5')
    # if (DATA_PATH / 'macro.h5').exists():
    #     is_recession = macrostore['USARECDM']
    # else:
    #     is_recession = DataReader('USARECDM', data_source='fred', start='1900')
    #     is_recession = is_recession.resample('B').interpolate()
    #     is_recession.dropna(inplace=True)  # drops the first value which is nan
    #     macrostore['USARECDM'] = is_recession
    # macrostore.close()
    #
    # rf_feature = rf['DGS1MO_annual'].to_numpy()
    # rf_feature = rf_feature[-T:]
    # rf_feature = rf_feature[np.newaxis, :, np.newaxis]
    # rf_feature = np.repeat(rf_feature, N, axis=0)
    # riskless_rate = tf.Variable(rf_feature)
    # new_X = tf.concat([X, riskless_rate], axis=2)


    # datatf = tf.Variable(data)
    # datatf = tf.transpose(datatf)
    # datatf = tf.Variable(tf.reshape(datatf, [N, T, 1]))

    # multidata = data.stack().to_frame(name='prices')
    # idx = pd.IndexSlice
    # p = multidata.loc[idx[:, ['AAPL']], ['prices']]

    # mulidx = pd.MultiIndex.from_product([data.index, data.columns], names=['Date', 'Ticker'])
    # # False:
    # muldata_reconstitution = pd.DataFrame(X.numpy().reshape(-1, 6), index=mulidx)

    # # Extract features
    # get_level_1 = multidata.prices.groupby(level=1)
    # multidata['returns1p'] = get_level_1.pct_change()
    # multidata['returns2p'] = get_level_1.pct_change(2)
    # multidata['returns5p'] = get_level_1.pct_change(5)
    # multidata['returns10p'] = get_level_1.pct_change(10)
    # multidata['returns21p'] = get_level_1.pct_change(21)
    #
    # multidata2 = data.T.stack().to_frame(name='prices')
    # get_level_0 = multidata2.prices.groupby(level=0)
    # multidata2['returns1p'] = get_level_0.pct_change()
    # multidata2['returns2p'] = get_level_0.pct_change(2)
    # multidata2['returns5p'] = get_level_0.pct_change(5)
    # multidata2['returns10p'] = get_level_0.pct_change(10)
    # multidata2['returns21p'] = get_level_0.pct_change(21)
    #
    # D: int = len(multidata2.columns)
    # # TODO: wrong
    # datatf2 = tf.Variable(multidata2.unstack(level=1).values.reshape(N, T, D))
    #
    # sub = data.loc['2010-01-04':'2010-01-07', ['AAPL', 'MSFT']].copy()
    # stacksub = sub.T.stack().to_frame(name='prices')
    # stacksub['returns1p'] = stacksub.prices.groupby(level=0).pct_change()
    # stacksub['returns2p'] = stacksub.prices.groupby(level=0).pct_change(2)
    # subtf = tf.Variable(stacksub)
    # subtfreshaped = tf.reshape(subtf, [2, 4, 3])

    # Add macro-indicators, including the risk-free rate
