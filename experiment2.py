import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner



def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "rchen613"  # replace tb34 with your Georgia Tech username


def generate_experiment_2(symbol, sd, ed, start_val, impact):
    sl = StrategyLearner(impact=impact, commission=0)
    sl.add_evidence(symbol="JPM", sd=sd, ed=ed)
    in_sample_trades = sl.testPolicy(symbol="JPM", sd=sd, ed=ed)
    orders_df = pd.DataFrame(index=in_sample_trades.index, columns=['Symbol', 'Order', 'Shares'])
    orders_df['Symbol'] = symbol
    orders_df['Order'] = np.select([(in_sample_trades.iloc[:, 0] > 0),(in_sample_trades.iloc[:, 0] < 0)], ['BUY', 'SELL'], default='HOLD')
    orders_df['Shares'] = np.abs(in_sample_trades.iloc[:, 0])
    sl_portval = compute_portvals(orders_df, start_val=start_val) / start_val
    return sl_portval







