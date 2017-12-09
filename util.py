from os.path import join, dirname

from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt


def symbol_to_path(symbol):
    """Return CSV file path given ticker symbol."""
    return join(dirname(__file__), "data", "{}.csv".format(str(symbol)))


def get_data(symbols, dates, bench_sym, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = DataFrame(index=dates)
    if addSPY and bench_sym not in symbols:  # add SPY for reference, if absent
        symbols = [bench_sym] + symbols

    for symbol in symbols:
        df_temp = read_csv(symbol_to_path(symbol=symbol), 
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"], 
            index_col="Date_Time", parse_dates=[[0, 1]], na_values=["nan"])
        df_temp[symbol] = df_temp.Close
        df = df.join(df_temp[symbol])
        if symbol == bench_sym:
            df = df.dropna(subset=[bench_sym])

    return df


def plot_data(df, title="Prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
