from datetime import timedelta
from random import seed

from numpy import asarray
from pandas import DataFrame, date_range, cut
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import style

import qlearner as ql
from util import get_data

style.use('ggplot')
seed(0)


class Trade(object):
    class Action:
        LONG = 0
        SHORT = 1
        NOTHING = 2

    def __init__(self, symbol, start_date, end_date, investment, bench_sym, verbose=False):
        self.symbol = symbol
        self.investment = investment
        self.verbose = verbose
        self.cash = investment
        self.shares = 0
        self.position = 0
        self.bench_sym = bench_sym

        # Read data
        dates = date_range(start_date - timedelta(10), end_date)
        df = get_data(symbols=[symbol], dates=dates, bench_sym=self.bench_sym)[symbol]

        # Normalize close to starting date
        norm_val = get_data(symbols=[symbol], dates=date_range(start_date, start_date + timedelta(1)), bench_sym=self.bench_sym)[symbol]
        normed = df / norm_val.ix[0]

        # Determine features
        # Daily Returns
        returns = DataFrame(df)
        daily_returns = returns[symbol].copy()
        daily_returns[1:] = (returns[symbol].ix[1:] / returns[symbol].ix[:-1].values) - 1
        daily_returns.ix[0] = 0

        # Relative Strnength Index
        up, down = daily_returns.copy(), daily_returns.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean().abs()
        rsi = 100.0 - (100.0 / (1.0 + roll_up / roll_down))

        # Simple moving average
        sma5 = normed.rolling(5).mean()
        sma10 = normed.rolling(10).mean()
        sma15 = normed.rolling(15).mean()
        sma20 = normed.rolling(20).mean()
        sma25 = normed.rolling(25).mean()
        sma30 = normed.rolling(30).mean()
        sma40 = normed.rolling(40).mean()

        # Volatility
        vol5 = normed.rolling(5).std()
        vol10 = normed.rolling(10).std()
        vol20 = normed.rolling(20).std()
        vol30 = normed.rolling(30).std()

        # Bollinger bands
        sma_bb = normed.rolling(5).mean()
        sma_bb_std = normed.rolling(5).std()
        bb = (normed - (sma_bb - 2 * sma_bb_std)) / ((sma_bb + 2 * sma_bb_std) - (sma_bb - 2 * sma_bb_std))

        # Moving average convergence/divergence
        ema12 = normed.ewm(span=12).mean()
        ema26 = normed.ewm(span=26).mean()

        macd = ema12 - ema26

        # Momentum
        momentum2 = normed / normed.shift(2) - 1
        momentum5 = normed / normed.shift(5) - 1
        momentum10 = normed / normed.shift(10) - 1

        # Combine into new dataframe
        df = DataFrame(df) \
            .assign(sma15=normed / sma15 - 1).assign(sma5=normed / sma5 - 1) \
            .assign(bb=bb).assign(rsi=rsi) \
            .assign(momentum2=momentum2) \
            .assign(normed=normed) \
            .assign(macd=macd) \
            .assign(vol10=vol10).assign(vol20=vol20) \
            .assign(vol30=vol30) \
            .assign(vol10=vol10) \
            .assign(sma10=normed / sma10 - 1) \
            .assign(sma20=normed / sma20 - 1) \
            .assign(sma25=normed / sma25 - 1) \
            .assign(sma30=normed / sma30 - 1) \
            .assign(sma40=normed / sma40 - 1) \
            .assign(vol5=vol5).assign(momentum5=momentum5) \
            .assign(momentum10=momentum10)[start_date:]
        daily_returns.ix[0] = 0
        df = df.assign(dr=daily_returns)

        # Determine optimal features for states
        corr_df = df.corr().abs()
        corr = corr_df['dr'][:]
        icorr = asarray(corr)
        scorr = icorr.argsort()[-4:][::-1]  # select top 3 features and daily returns
        scorr = scorr[1:]  # remove daily returns from possible features

        optimal_features = []
        for i in scorr:
            optimal_features.append(corr_df.columns[i])

        self.optimal_features = optimal_features
        self.df = df
        self.market = df.iterrows()
        self.current = next(self.market)
        self.action = self.Action()

    def buy(self):
        self.shares = 100
        close = self.current[1][self.symbol]
        if self.position == 1:
            return -50
        elif self.position == 0:
            self.position = 1
            if self.verbose:
                print("Buy {} contracts at $%0.2f".format(self.shares, close))
            return 10 * self.current[1]['dr']
        elif self.position == 2:
            self.position = 1
            if self.verbose:
                print("Close position and buy {} ontracts at $%0.2f".format(self.shares, close))
            return 10 * self.current[1]['dr']
        else:
            self.position = 1
            print("Error: position unknown")
            return -50

    def sell(self):
        self.shares = -100
        close = self.current[1][self.symbol]
        if self.position == 1:
            self.position = 2
            if self.verbose:
                print("Close position and short {} contracts at $%0.2f".format(self.shares, close))
            return -10 * self.current[1]['dr']
        elif self.position == 0:
            self.position = 2
            if self.verbose:
                print("Short {} contract at $%0.2f".format(self.shares, close))
            return -10 * self.current[1]['dr']
        elif self.position == 2:
            return -50
        else:
            self.position = 2
            print("Error: position unknown")
            return -50

    def hold(self):
        if self.position == 1:
            if self.verbose:
                print("Hold long position")
            return 5 * self.current[1]['dr']
        elif self.position == 0:
            if self.verbose:
                print("Hold cash position")
            return 0
        elif self.position == 2:
            return -5 * self.current[1]["dr"]
        else:
            self.position = 0
            print("Error: position unknown")
            return -100

    def discretize(self):
        date = self.current[0]
        s = self.position
        for i, feature in enumerate(self.optimal_features):
            s += (10 ** (i + 1)) * cut(self.df[feature], 1, labels=False)[date]

        return int(s)

    def reward(self, action):
        # Calculate reward from given action
        r = {self.action.LONG: self.buy, self.action.SHORT: self.sell, self.action.NOTHING: self.hold, }[action]()

        # Find state of next day
        try:
            self.current = next(self.market)
            state = self.discretize()
        except StopIteration:
            return None, None

        return state, r

    def state(self):
        close = self.current[1][self.symbol]
        value = self.shares * self.current[1]["dr"] + self.cash
        return value, self.cash, self.shares, close

    def baseline(self):
        return (self.df[self.symbol].ix[-1] - self.df[self.symbol].ix[0]) * 100

    def rawData(self):
        return self.df


class StrategyLearner(object):
    def __init__(self, bench_sym, verbose=False):
        self.verbose = verbose
        self.bench_sym = bench_sym
        self.ql = ql.QLearner(num_states=int(1e6), num_actions=3, alpha=0.1, gamma=0.9, rar=0.5, radr=0.9, dyna=0, verbose=False)

    # this method should create a QLearner, and train it for trading
    def add_evidence(self, symbol, start_date, end_date, investment):
        returns = -1  # current return
        i = 0  # loop iterator
        while i < 10:
            i += 1
            trade = Trade(symbol=symbol, start_date=start_date, end_date=end_date, investment=investment, bench_sym=self.bench_sym, verbose=self.verbose)
            s = trade.discretize()
            a = self.ql.querysetstate(s)
            while True:
                s1, r = trade.reward(a)
                if s1 is None:
                    break
                a = self.ql.query(s1, r)

            returns0 = returns
            returns = trade.state()[0]
            if (returns == returns0) & (i > 200):
                break

            if i > 1000:
                print("Error: cannot converge")
                break

    # this method should use the existing policy and test it against new data
    def test_policy(self, symbol, start_date, end_date, investment):
        trade = Trade(symbol=symbol, start_date=start_date, end_date=end_date, investment=investment, bench_sym=self.bench_sym, verbose=self.verbose)
        s = trade.discretize()
        a = self.ql.querysetstate(s)

        df = trade.rawData()
        actions = []

        while True:
            s1, r = trade.reward(a)
            if s1 is None:
                break
            a = self.ql.querysetstate(s1)
            actions.append(a)
            if self.verbose:
                print(s1, r, a)

        # Correct for holding when trying to long/short above leverage maximum
        prev = 0  # initialize previous action as buy
        actions = asarray([0] + actions)
        for i in range(1, len(actions) - 1):
            if actions[i] != 2:
                if actions[i] == prev:
                    actions[i] = 2
                else:
                    prev = actions[i]

        actions[-1] = 1  # sell on last day
        df['Trades'] = actions

        def order(x):
            if x == 0:
                return 'BUY'
            elif x == 1:
                return 'SELL'
            else:
                return 0

        df = df[df["Trades"] != 2].copy()
        df["Order"] = df["Trades"].apply(lambda x: order(x))
        df["Shares"] = 200
        df["Shares"].ix[0] = 100
        df = df[["Order", "Shares"]].copy()

        # Create benchmark dataframe
        start = df.index[0]
        end = df.index[-1]
        benchmark = DataFrame({"Order": ["BUY", "SELL"], "Shares": [100, 100]}, index=[start, end])
        return df, benchmark


if __name__ == "__main__":
    pass
