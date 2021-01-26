from __future__ import annotations
import numpy as np


def check_type(o, classes: list):
    """Checks whether the object `o` is an instance is at
    least part of one of the `classes`. Otherwise, raise
    a TypeError with the right debug message."""
    bools: list = [isinstance(o, c) for c in classes]
    class_names_li: list[str] = [f"{c.__name__!r}" if isinstance(c, type) else f"{type(c).__name__!r}" for c in classes]
    class_names: str = ', or '.join(class_names_li)
    if sum(bools) == 0:
        raise TypeError(f"Object of type {class_names} expected. But got {type(o).__name__!r} instead.")


def my_ceil(arr, precision: int = 0):
    return np.round(arr + 0.5 * 10**(-precision), precision)


def my_floor(arr, precision: int = 0):
    return np.round(arr - 0.5 * 10**(-precision), precision)


# def currency_exchange(eur: Annotated[float, "euros"],
#                       rate: Annotated[float, "exchange rate"]) -> Annotated[float, "euro to dollars"]:
#     """Converting Euros to Dollars using the exchange rate"""
#     return eur * rate


def track_results(episode, nav_ma_100, nav_ma_10,
                  market_nav_100, market_nav_10,
                  win_ratio, total, epsilon, episode_time):
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)

    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    print(template.format(episode, format_time(total),
                          nav_ma_100 - 1, nav_ma_10 - 1,
                          market_nav_100 - 1, market_nav_10 - 1,
                          win_ratio, epsilon))


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


if __name__ == '__main__':
    b = 1
    # TypeError: Object of type 'float', or 'str' expected. But got 'int' instead.
    check_type(b, [float, str])
