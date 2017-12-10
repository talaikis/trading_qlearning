from datetime import datetime

from pandas import date_range, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
from numpy import asarray

import strategy_learner as sl
from util import get_data

style.use('ggplot')


def run_algo(sym, investment, start_date, end_date, bench_sym):
    # instantiate the strategy learner
    learner = sl.StrategyLearner(bench_sym=bench_sym, verbose=verbose)

    # train the learner
    learner.add_evidence(symbol=sym, start_date=start_date, end_date=end_date, investment=investment)

    # get some data for reference
    syms = [sym]
    dates = date_range(start_date, end_date)
    prices_all = get_data(symbols=syms, dates=dates, bench_sym=bench_sym)
    prices = prices_all[syms]

    # test the learner
    df_trades = learner.test_policy(symbol=sym, start_date=start_date, end_date=end_date, investment=investment)

    return df_trades


def evaluate(sym, orders, start_val, fee, slippage, bench_sym):
    # Read orders file
    orders_df = orders

    orders_df.sort_index(inplace=True)
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]

    # Collect price data for each ticker in order
    df_prices = get_data(symbols=[sym], dates=date_range(start_date, end_date), bench_sym=bench_sym)
    df_prices = df_prices.drop(bench_sym, 1)
    df_prices["cash"] = 1

    # Track trade data
    df_trades = df_prices.copy()
    df_trades[:] = 0

    # Populate trade dataframe
    for i, date in enumerate(orders_df.index):
        # Get order information
        if orders_df.Order[i] == "BUY":
            order = 1
        else:
            order = -1

        # Start with 1/2 position at first
        if i == 0:
            shares = 100
        else:
            shares = 200

        # Calculate change in shares and cash
        df_trades[sym][date] += order * shares
        df_trades['cash'][date] -= order * (1 - slippage) * shares * df_prices[sym][date] - fee

    # Track total holdings
    df_holdings = df_prices.copy()
    df_holdings[:] = 0

    # Include starting value
    df_holdings['cash'][0] = start_val

    # Update first day of holdings
    for c in df_trades.columns:
        df_holdings[c][0] += df_trades[c][0]

    # Update every day, adding new day's trade information with previous day's holdings
    for i in range(1, len(df_trades.index)):
        for c in df_trades.columns:
            df_holdings[c][i] += df_trades[c][i] + df_holdings[c][i - 1]

    # Track monetary values
    df_values = df_prices.mul(df_holdings)

    # Define port_val
    port_val = df_values.sum(axis=1)

    return port_val


if __name__ == "__main__":
    symbol = "NASDAQ1001440"
    bench_sym = "S&P5001440"
    verbose = False
    investment = 100000
    fee = 0
    slippage = 0.0025  # in %
    start_date_insample = datetime(2013, 5, 1)
    end_date_insample = datetime(2015, 5, 1)
    start_date_outsample = datetime(2015, 5, 2)
    end_date_outsample = datetime(2017, 12, 7)

    # Train
    df_trades_in, benchmark_in = run_algo(sym=symbol, investment=investment, start_date=start_date_insample, end_date=end_date_insample, bench_sym=bench_sym)
    df_trades_out, benchmark_out = run_algo(sym=symbol, investment=investment, start_date=start_date_outsample, end_date=end_date_outsample, bench_sym=bench_sym)

    # Evaluate
    insample = evaluate(sym=symbol, orders=df_trades_in, start_val=investment, fee=fee, slippage=slippage, bench_sym=bench_sym)
    insample = DataFrame(insample)
    bench_insample = evaluate(sym=symbol, orders=benchmark_in, start_val=investment, fee=fee, slippage=slippage, bench_sym=bench_sym)
    bench_insample = DataFrame(bench_insample)
    outsample = evaluate(sym=symbol, orders=df_trades_out, start_val=investment, fee=fee, slippage=slippage, bench_sym=bench_sym)
    outsample = DataFrame(outsample)
    bench_outsample = evaluate(sym=symbol, orders=benchmark_out, start_val=investment, fee=fee, slippage=slippage, bench_sym=bench_sym)
    bench_outsample = DataFrame(bench_outsample)

    # Cumulative returns
    port_ret_in = float(asarray(insample.values)[-1])
    port_ret_out = float(asarray(outsample.values)[-1])
    bench_ret_in = float(asarray(bench_insample.values)[-1])
    bench_ret_out = float(asarray(bench_outsample.values)[-1])

    # Print results
    print()
    print("Cumulative return in-sample:\t\t${:,.2f}\t\t(+{:.2f} %)".format(port_ret_in - investment, 100 * (port_ret_in - investment) / investment))
    print("Benchmark return in-sample:\t\t\t${:,.2f}\t\t(+{:.2f} %)".format(bench_ret_in - investment, 100 * (bench_ret_in - investment) / investment))
    print("Cumulative return out-of-sample:\t${:,.2f}\t\t(+{:.2f} %)".format(port_ret_out - investment, 100 * (port_ret_out - investment) / investment))
    print("Benchmark return out-of-sample:\t\t${:,.2f}\t\t(+{:.2f} %)".format(bench_ret_out - investment, 100 * (bench_ret_out - investment) / investment))

    # Plot charts
    plt.subplot(1, 2, 1)
    plt.plot(insample.index, insample, c="mediumseagreen", lw=3)
    plt.plot(bench_insample.index, bench_insample, c="skyblue")
    plt.legend(["Strategy", "Buy and Hold"])
    plt.title("In-sample")
    plt.xlabel("Date")
    plt.ylabel("Value")

    plt.subplot(1, 2, 2)
    plt.plot(outsample.index, outsample, c="mediumseagreen", lw=3)
    plt.plot(bench_outsample.index, bench_outsample, c="skyblue")
    plt.legend(["Strategy", "Buy and Hold"])
    plt.title("Out-of-sample")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()
