import pandas as pd

# Indicator 5
def momentum(norm_prices, rolling_days):
    momentum = pd.DataFrame(0, index=norm_prices.index, columns=['momentum'])
    momentum['momentum'] = norm_prices.diff(rolling_days)/norm_prices.shift(rolling_days)
    buy_quantile = momentum.quantile(0.9).values[0]
    sell_quantile = momentum.quantile(0.1).values[0]
    momentum['buy_signal'] = momentum['momentum'] > buy_quantile
    momentum['sell_signal'] = momentum['momentum'] < sell_quantile
    momentum['action'] = -momentum['sell_signal'].astype(int) + momentum['buy_signal'].astype(int)
    return momentum


def exponential_moving_average(norm_prices, rolling_days):
    return norm_prices.ewm(span=rolling_days, adjust=False, min_periods=rolling_days).mean()


# Indicator 1
def percentage_price_oscillator(norm_prices, rolling_days1, rolling_days2):
    ppo = pd.DataFrame(0, index=norm_prices.index, columns=['ppo', 'signal', 'diff'])
    ppo['rolling1_EMA'] = norm_prices.ewm(span=rolling_days1).mean()
    ppo['rolling2_EMA'] = norm_prices.ewm(span=rolling_days2).mean()
    ppo['ppo'] = (ppo['rolling1_EMA'] - ppo['rolling2_EMA']) / ppo['rolling2_EMA'] * 100
    ppo['signal'] = exponential_moving_average(ppo['ppo'], 9)
    ppo['diff'] = ppo['ppo'] - ppo['signal']
    ppo['sell_signal'] = (ppo['diff'] < 0) & (ppo['diff'].shift() > 0)
    ppo['buy_signal'] = (ppo['diff'] > 0) & (ppo['diff'].shift() < 0)
    ppo['action'] = -ppo['sell_signal'].astype(int) + ppo['buy_signal'].astype(int)
    return ppo.dropna()


# Indicator 2
def MACD(norm_prices, rolling_days1, rolling_days2):
    macd = pd.DataFrame(0, index=norm_prices.index, columns=['EMA1', 'EMA2', 'macd'])
    macd['EMA1'] = norm_prices.ewm(span=rolling_days1).mean()
    macd['EMA2'] = norm_prices.ewm(span=rolling_days2).mean()
    macd['macd'] = macd['EMA1'] - macd['EMA2']
    macd['signal'] = macd["macd"].ewm(span=9).mean()

    macd['diff'] = macd['macd'] - macd['signal']
    macd['sell_signal'] = (macd['diff'] < 0) & (macd['diff'].shift() > 0)
    macd['buy_signal'] = (macd['diff'] > 0) & (macd['diff'].shift() < 0)
    macd['action'] = -macd['sell_signal'].astype(int) + macd['buy_signal'].astype(int)
    return macd.dropna()


# Indicator 3
def simple_moving_average(norm_prices, rolling_days):
    sma = pd.DataFrame(0, index=norm_prices.index, columns=['SMA'])
    sma['SMA'] = norm_prices.rolling(window=rolling_days).mean()
    sma['price/SMA'] = norm_prices.iloc[:,0]/sma['SMA']
    sma['price'] = norm_prices
    sma['buy_signal'] = (sma['price/SMA'] > 1) & (sma['price/SMA'].shift() < 1)
    sma['sell_signal'] = (sma['price/SMA'] < 1) & (sma['price/SMA'].shift() > 1)
    sma['action'] = -sma['sell_signal'].astype(int) + sma['buy_signal'].astype(int)
    return sma

# indicator 4
def bollinger_band(norm_prices, rolling_days):
    bollinger_band = pd.DataFrame(index=norm_prices.index)
    bollinger_band['rolling_std'] = norm_prices.rolling(window=rolling_days, min_periods=rolling_days).std()
    bollinger_band['sma'] = norm_prices.rolling(window=rolling_days).mean()
    bollinger_band['top_band'] = bollinger_band['sma'] + 2 * bollinger_band['rolling_std']
    bollinger_band['bottom_band'] = bollinger_band['sma'] - 2 * bollinger_band['rolling_std']
    bollinger_band['indicator'] = (norm_prices.iloc[:,0] - bollinger_band['sma']) / (2*bollinger_band['rolling_std'])
    bollinger_band['sell_signal'] = (bollinger_band['indicator'] > 1) & (bollinger_band['indicator'].shift() < 1)
    bollinger_band['buy_signal'] = (bollinger_band['indicator'] < -1) & (bollinger_band['indicator'].shift() > -1)
    bollinger_band['action'] = -bollinger_band['sell_signal'].astype(int) + bollinger_band['buy_signal'].astype(int)
    return bollinger_band

