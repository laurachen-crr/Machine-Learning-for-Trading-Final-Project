""""""  		  	   		 	   			  		 			     			  	 
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
GT User ID: rchen613 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903971253 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt  		  	   		 	   			  		 			     			  	 
import indicators
import pandas as pd  		  	   		 	   			  		 			     			  	 
import util as ut  		  	   		 	   			  		 			     			  	 
from datetime import datetime, timedelta


class ManualStrategy(object):
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
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.verbose = verbose  		  	   		 	   			  		 			     			  	 
        self.impact = impact  		  	   		 	   			  		 			     			  	 
        self.commission = commission
        self.indicator_data = pd.DataFrame()
        self.look_back_period = {'SMA': 20, 'MACD': [12, 26], 'PPO': [12, 26], 'BBP': 35, 'momentum': 30}


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "rchen613"  # replace tb34 with your Georgia Tech username


    # this method should create a QLearner, and train it for trading  		  	   		 	   			  		 			     			  	 
    def add_evidence(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 	   			  		 			     			  	 
        sv=10000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        add evidence is used here for consistency with the Strategy Learner. In actuality, it is likely that 
        this function does nothnig as the "rules" will have been coded by the developer of the class. If that is the case,
        this this method might simply consist of a "pass" statement.

        Trains your strategy learner over a given time frame.

        Parameters
            symbol (str) – The stock symbol to train on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) – The starting value of the portfolio  	   		 	   			  		 			     			  	 
        """

        # add your code to do learning here
        # example usage of the old backward compatible util function  		  	   		 	   			  		 			     			  	 
        syms = [symbol]  		  	   		 	   			  		 			     			  	 
        dates = pd.date_range(sd-timedelta(days=30), ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
        prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 
        if self.verbose:
            print(prices)


    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			     			  	 
    def testPolicy(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
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
        dates = pd.date_range(sd-timedelta(days=30), ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
        prices = prices_all[[symbol,]]  # only portfolio symbols

        normed_prices = prices / prices.iloc[0]
        self.indicator_data['SMA'] = indicators.simple_moving_average(normed_prices, self.look_back_period['SMA'])['action']
        self.indicator_data['MACD'] = indicators.MACD(normed_prices, self.look_back_period['MACD'][0], self.look_back_period['MACD'][1])['action']
        self.indicator_data['PPO'] = indicators.percentage_price_oscillator(normed_prices, self.look_back_period['PPO'][0], self.look_back_period['PPO'][1])['action']
        self.indicator_data['BBP'] = indicators.bollinger_band(normed_prices, self.look_back_period['BBP'])['action']
        self.indicator_data['momentum'] = indicators.momentum(normed_prices, self.look_back_period['momentum'])['action']

        actions = pd.DataFrame(columns=['actions'])
        actions['actions'] = self.indicator_data['SMA'] + self.indicator_data['MACD'] + self.indicator_data['PPO']
        actions = actions[sd:ed]
        trades = pd.DataFrame(columns=['pos'], index=actions.index)
        position = 0
        for idx, action in actions.iterrows():
            action = action['actions']
            if action > 2:
                if position == 0:
                    trades.loc[idx] = 1000
                elif position == -1000:
                    trades.loc[idx] = 2000
                else:
                    trades.loc[idx] = 0
            elif action < -2:
                if position == 0:
                    trades.loc[idx] = -1000
                elif position == 1000:
                    trades.loc[idx] = -2000
                else:
                    trades.loc[idx] = 0
            else:
                trades.loc[idx] = 0
            position += trades.loc[idx].iloc[0]
        return trades


if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    print("One does not simply think up a strategy")
