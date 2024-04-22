import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from matplotlib import pyplot as plt
import util as ut


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "rchen613"  # replace tb34 with your Georgia Tech username


def plot_experiment_1(symbol, sl_trades, sd, ed, start_val, file_name):
    ms = ManualStrategy()
    trades = ms.testPolicy(symbol="JPM", sd=sd, ed=ed)
    orders_df = pd.DataFrame(index=trades.index, columns=['Symbol', 'Order', 'Shares'])
    orders_df['Symbol'] = symbol
    orders_df['Order'] = np.select([(trades.iloc[:, 0] > 0),(trades.iloc[:, 0] < 0)], ['BUY', 'SELL'], default='HOLD')
    orders_df['Shares'] = np.abs(trades.iloc[:, 0])
    manual_portval = compute_portvals(orders_df, start_val=start_val) / start_val
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
    prices = prices_all[[symbol, ]]  # only portfolio symbols
    benchmark = (prices * 1000 + (start_val - prices.iloc[0, 0] * 1000)) / start_val

    orders_df_2 = pd.DataFrame(index=sl_trades.index, columns=['Symbol', 'Order', 'Shares'])
    orders_df_2['Symbol'] = symbol
    orders_df_2['Order'] = np.select([(sl_trades.iloc[:, 0] > 0),(sl_trades.iloc[:, 0] < 0)], ['BUY', 'SELL'], default='HOLD')
    orders_df_2['Shares'] = np.abs(sl_trades.iloc[:, 0])
    strategy_learner_portval = compute_portvals(orders_df_2, start_val=start_val) / start_val

    fig = plt.figure()  # create the top-level container
    plt.plot(manual_portval, label='Manual Strategy', color='red')
    plt.plot(benchmark, label='benchmark', color='purple')
    plt.plot(strategy_learner_portval, label='Strategy Learner', color='blue')

    plt.legend()
    plt.title(file_name)
    plt.xlabel("Date")
    plt.xticks(rotation=30)
    plt.ylabel("Portfolio Value")
    plt.savefig(f"./images/{file_name}.png", dpi=300)


