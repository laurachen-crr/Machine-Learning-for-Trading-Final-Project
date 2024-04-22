import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from matplotlib import pyplot as plt
import datetime as dt
import util as ut
from experiment2 import generate_experiment_2
from experiment1 import plot_experiment_1


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "rchen613"  # replace tb34 with your Georgia Tech username


def plot_manual_strategy(symbol, sd, ed, start_val, file_name):
    ms = ManualStrategy(verbose=True)
    trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed)
    orders_df = pd.DataFrame(index=trades.index, columns=['Symbol', 'Order', 'Shares'])
    orders_df['Symbol'] = symbol
    orders_df['Order'] = np.select([(trades.iloc[:, 0] > 0),(trades.iloc[:, 0] < 0)], ['BUY', 'SELL'], default='HOLD')
    orders_df['Shares'] = np.abs(trades.iloc[:, 0])
    manual_portval = compute_portvals(orders_df, start_val=start_val) / start_val

    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
    prices = prices_all[[symbol, ]]  # only portfolio symbols
    benchmark = (prices * 1000 + (start_val - prices.iloc[0, 0] * 1000)) / start_val

    fig = plt.figure()  # create the top-level container
    plt.plot(manual_portval, label='Manual Strategy', color='red')
    plt.plot(benchmark, label='benchmark', color='purple')
    for idx, trade in trades.iterrows():
        if trade.iloc[0] > 0:
            plt.vlines(x=idx, ymin=0.8, ymax=1.4, color='blue')
        elif trade.iloc[0] < 0:
            plt.vlines(x=idx, ymin=0.8, ymax=1.4, color='black')
    plt.legend()
    plt.title(file_name)
    plt.xlabel("Date")
    plt.xticks(rotation=30)
    plt.ylabel("Portfolio Value")
    plt.savefig(f"./images/{file_name}.png", dpi=300)


if __name__ == "__main__":
    np.random.seed(903971253)
    symbol = "JPM"

    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)

    out_of_sample_sd = dt.datetime(2010, 1, 1)
    out_of_sample_ed = dt.datetime(2011, 12, 31)

    # -------------------- Manual Strategy -------------------
    plot_manual_strategy(symbol, in_sample_sd, in_sample_ed, 100000, "In-Sample Manual Strategy")
    plot_manual_strategy(symbol, out_of_sample_sd, out_of_sample_ed, 100000, "Out-of-Sample Manual Strategy")

    # --------------------- Experiment 1 ---------------------
    sl = StrategyLearner()
    sl.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed)
    in_sample_trades = sl.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed)
    out_of_sample_trades = sl.testPolicy(symbol=symbol, sd=out_of_sample_sd, ed=out_of_sample_ed)

    in_sample_text = ("Random Forest Bagging \n"
                      "In-Sample Benchmark v.s Manual v.s Strategy Learner \n"
                      "leaf_size=15, bag=20")
    out_of_sample_text = ("Random Forest Bagging \n "
                          "Out-of-Sample Benchmark v.s Manual v.s Strategy Learner \n"
                          "leaf_size=15, bag=20")
    plot_experiment_1(symbol, in_sample_trades, in_sample_sd, in_sample_ed, 100000, in_sample_text)
    plot_experiment_1(symbol, out_of_sample_trades, out_of_sample_sd, out_of_sample_ed, 100000, out_of_sample_text)

    dates = pd.date_range(in_sample_sd, out_of_sample_ed)
    prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
    prices = prices_all[[symbol, ]]  # only portfolio symbols
    benchmark = (prices * 1000 + (100000 - prices.iloc[0, 0] * 1000)) / 100000

    # --------------------- Experiment 2 ---------------------
    impact = [0.005, 0.010, 0.015]
    sl1_portval = generate_experiment_2(symbol, in_sample_sd, in_sample_ed, 100000, impact[0])
    sl2_portval = generate_experiment_2(symbol, in_sample_sd, in_sample_ed, 100000, impact[1])
    sl3_portval = generate_experiment_2(symbol, in_sample_sd, in_sample_ed, 100000, impact[2])

    sl1_daily_return = (sl1_portval - sl1_portval.shift(1)) / sl1_portval
    sl2_daily_return = (sl2_portval - sl2_portval.shift(1)) / sl2_portval
    sl3_daily_return = (sl3_portval - sl3_portval.shift(1)) / sl3_portval

    with open("result.txt", "w") as f:
        f.write(f"sl1 mean: {np.mean(sl1_daily_return)}\n")
        f.write(f"sl1 std: {np.std(sl1_daily_return)}\n")
        f.write(f"sl2 mean: {np.mean(sl2_daily_return)}\n")
        f.write(f"sl2 std: {np.std(sl2_daily_return)}\n")
        f.write(f"sl3 mean: {np.mean(sl3_daily_return)}\n")
        f.write(f"sl3 std: {np.std(sl3_daily_return)}\n")

    fig = plt.figure()
    plt.plot(sl1_portval, label=f"impact={impact[0]}")
    plt.plot(sl2_portval, label=f"impact={impact[1]}")
    plt.plot(sl3_portval, label=f"impact={impact[2]}")
    plt.title("In-Sample Study of Different Impact Values")
    plt.xlabel("Date")
    plt.xticks(rotation=30)
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.savefig("./images/In-Sample Study of Different Impact Values.png", dpi=300)
