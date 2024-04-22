""""""
import numpy as np

"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt  		  	   		 	   			  		 			     			  	 
import indicators
import pandas as pd
import util as ut
from datetime import datetime, timedelta
from QLearner import QLearner

class StrategyLearner(object):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # constructor  		  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.verbose = verbose  		  	   		 	   			  		 			     			  	 
        self.impact = impact  		  	   		 	   			  		 			     			  	 
        self.commission = commission
        self.num_bins = 10
        self.bin_edges = pd.DataFrame()
        self.data = pd.DataFrame(columns=['state', 'action', 'trades'])
        self.ql = QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)
        self.num_epochs = 500
        self.look_back_period = {'SMA': 20, 'MACD': [12, 26], 'PPO': [12, 26], 'BBP': 35, 'momentum': 30}
        self.delta_days = 100


    def create_bin_edges(self, indicator_data):
        for indicator in indicator_data:
            hist, bin_edges = np.histogram(indicator_data[indicator])
            self.bin_edges[indicator] = bin_edges


    def discretize_indicators(self, indicator_dict):
        sum_discrete = 0
        for indicator_name, value in indicator_dict.items():
            discretized_value = 0
            for k in self.bin_edges[indicator_name]:
                if value < k:
                    break
                else:
                    discretized_value += 1
            sum_discrete += discretized_value
        return int(sum_discrete)


    # this method should create a QLearner, and train it for trading  		  	   		 	   			  		 			     			  	 
    def add_evidence(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2009, 12, 31),
        sv=10000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        # add your code to do learning here  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        # example usage of the old backward compatible util function  		  	   		 	   			  		 			     			  	 
        syms = [symbol]  		  	   		 	   			  		 			     			  	 
        dates = pd.date_range(sd-timedelta(days=self.delta_days), ed+timedelta(days=self.delta_days))
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        normed_prices = prices / prices.iloc[0]

        train_indicator_data = pd.DataFrame()
        train_indicator_data['SMA'] = indicators.simple_moving_average(normed_prices, self.look_back_period['SMA'])['price/SMA']
        train_indicator_data['MACD'] = indicators.MACD(normed_prices, self.look_back_period['MACD'][0], self.look_back_period['MACD'][1])['diff']
        train_indicator_data['PPO'] = indicators.percentage_price_oscillator(normed_prices, self.look_back_period['PPO'][0], self.look_back_period['PPO'][1])['diff']
        train_indicator_data['BBP'] = indicators.bollinger_band(normed_prices, self.look_back_period['BBP'])['indicator']
        train_indicator_data['momentum'] = indicators.momentum(normed_prices, self.look_back_period['momentum'])['momentum']

        train_indicator_data = train_indicator_data[sd:ed]

        self.create_bin_edges(train_indicator_data)
        daily_returns = ((prices-prices.shift(1))/prices.shift(1))[sd:ed]
        daily_returns.iloc[0] = 0

        self.data = pd.DataFrame(columns=['state', 'action', 'trades'], index=train_indicator_data.index)
        self.data['state'] = 0
        for idx in self.data.index:
            self.data.loc[idx, 'state'] = self.discretize_indicators(train_indicator_data.loc[idx].to_dict())

        trades = pd.DataFrame(columns=['pos'], index=daily_returns.index)
        for i in range(self.num_epochs):
            # initialize the state and action
            state = int(self.data['state'].iloc[0])
            self.data.loc[self.data.index[0], 'action'] = action = self.ql.querysetstate(state)
            position = 0
            for idx in self.data.index:
                reward = position * daily_returns.loc[idx] * (1 - self.impact) - self.commission
                action = self.ql.query(int(float(self.data.loc[idx]['state'])), reward)
                if action == 0:  # short
                    trades.loc[idx, 'pos'] = - position - 1000
                elif action == 1:  # flat
                    trades.loc[idx, 'pos'] = 0
                elif action == 2:  # long
                    trades.loc[idx, 'pos'] = - position + 1000
                else:
                    print("Wrong action!")
                position += trades.loc[idx, 'pos']
            if (trades['pos'] == self.data['trades']).all():
                self.data['trades'] = trades['pos']
                break
            else:
                self.data['trades'] = trades['pos']


    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			     			  	 
    def testPolicy(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=10000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        # here we build a fake set of trades  		  	   		 	   			  		 			     			  	 
        # your code should return the same sort of data  		  	   		 	   			  		 			     			  	 
        syms = [symbol]
        dates = pd.date_range(sd - timedelta(days=self.delta_days), ed + timedelta(days=self.delta_days))
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        normed_prices = prices / prices.iloc[0]

        train_indicator_data = pd.DataFrame()
        train_indicator_data['SMA'] = indicators.simple_moving_average(normed_prices, self.look_back_period['SMA'])['price/SMA']
        train_indicator_data['MACD'] = indicators.MACD(normed_prices, self.look_back_period['MACD'][0], self.look_back_period['MACD'][1])['diff']
        train_indicator_data['PPO'] = indicators.percentage_price_oscillator(normed_prices, self.look_back_period['PPO'][0], self.look_back_period['PPO'][1])['diff']
        train_indicator_data['BBP'] = indicators.bollinger_band(normed_prices, self.look_back_period['BBP'])['indicator']
        train_indicator_data['momentum'] = indicators.momentum(normed_prices, self.look_back_period['momentum'])['momentum']
        train_indicator_data = train_indicator_data[sd:ed]

        self.create_bin_edges(train_indicator_data)
        daily_returns = ((prices - prices.shift(1)) / prices.shift(1))[sd:ed]
        daily_returns.iloc[0] = 0

        self.data = pd.DataFrame(columns=['state', 'action', 'trades'], index=train_indicator_data.index)
        self.data['state'] = 0
        for idx in self.data.index:
            self.data.loc[idx, 'state'] = self.discretize_indicators(train_indicator_data.loc[idx].to_dict())

        trades = pd.DataFrame(columns=['pos'], index=daily_returns.index)
        position = 0
        for idx in self.data.index:
            reward = position * daily_returns.loc[idx] * (1 - self.impact)
            action = self.ql.query(int(float(self.data.loc[idx]['state'])), reward)
            if action == 0:  # short
                trades.loc[idx, 'pos'] = - position - 1000
            elif action == 1:  # flat
                trades.loc[idx, 'pos'] = 0
            elif action == 2:  # long
                trades.loc[idx, 'pos'] = - position + 1000
            else:
                print("Wrong action!")
            position += trades.loc[idx, 'pos']
        return trades


if __name__ == "__main__":
    print("One does not simply think up a strategy")  		  	   		 	   			  		 			     			  	 
    sl = StrategyLearner()
    sl.add_evidence("ML4T-220")
    trades = sl.testPolicy()
    print(trades)