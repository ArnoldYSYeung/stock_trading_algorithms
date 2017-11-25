##  Author: Arnold YS Yeung (contact@arnoldyeung.com)
##  Date:   November 25th, 2017
##  This script calculates technical time series features including rolling mean, rolling std,
##  rolling Sharpe Ratio, Bollinger Bands, and rolling Relative Strength Index (RSI)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import assessment as ass

def calculate_tech_feats(sd, ed, syms, rfr, rsi_window = 14, rolling_window = 50):
#   calculates technical features over time series

    dates = pd.date_range(sd, ed);  #   create date range
    price = ass.create_price_df(syms, dates, ref_opt = False);       #   adjusted close df
    norm_price = ass.norm_df(price, 0);             #   normalize price based on first index

    mean_price = norm_price.rolling(rolling_window).mean();
    std_price = norm_price.rolling(rolling_window).std();
    price_uband, price_lband = get_bollinger_bands(mean_price, std_price);
    RSI, RS = get_RSI(norm_price, rsi_window);              #   relative strength index

    dr, adr, sddr = ass.get_daily_returns(price);
    mean_dr = dr.rolling(rolling_window).mean();    #   rolling mean daily return
    std_dr = dr.rolling(rolling_window).std();      #   rolling standard deviation of daily return

    sr = get_rolling_sharpe_ratio(dr, rolling_window, rfr);     #   rolling sharpe ratio

    df_price = pd.concat([norm_price, mean_price, std_price, price_uband, price_lband], \
                         axis = 1);
    df_price.columns = ['Normalized Price', 'Mean Price', 'StD Price', 'Upper Band', 'Lower Band'];
    ass.plot_data(df_price, syms[0] + ' Price');

    df_daily_return = pd.concat([dr, mean_dr, std_dr], axis = 1);
    df_daily_return.columns = ['Daily Return', 'Mean Daily Return', 'StD Daily Return'];
    ass.plot_data(df_daily_return, syms[0] + ' Daily Return');

    ass.plot_data(RSI, 'Relative Strength Index');

    return 0;

def get_rolling_sharpe_ratio(daily_returns, rolling_window, rfr = 0.01, sf = 252):
#   calculate rolling sharpe ratio and output dataframe
#   default risk free rate = 0.01 per annum, sampling freq = 252 per annum
    sr = pd.DataFrame(None, index = daily_returns.index, columns = daily_returns.columns);   
    for i in range(rolling_window, len(sr.index)):
        #   add +1 because does not consider last element
        sr.iloc[i] = ass.get_sharpe_ratio(daily_returns.iloc[i - rolling_window:i+1], rfr, sf);
        #   Note: column header for dr and sr must be the same
    return sr

def get_bollinger_bands(mean, std):
#   calculate the upper and lower bollinger bands for the inputted mean dataframe
    upper_band = mean + 2*std;
    lower_band = mean - 2*std;
    return upper_band, lower_band;

def get_RSI(price, n = 14):
#   calculate the Relative Strength Index over n days (default 14)
#   the Relative Strength Index is a measure of momentum
    U = pd.DataFrame(None, index = price.index, columns = price.columns);
    D = pd.DataFrame(None, index = price.index, columns = price.columns);
    RS = pd.DataFrame(None, index = price.index, columns = price.columns);  #   relative strength
    RSI = pd.DataFrame(None, index = price.index, columns = price.columns);  #   relative strength index
    
    for i in range(1, len(price.index)):
        diff = float(price.iloc[i]) - float(price.iloc[i-1]);
        if diff > 0:            #   upward movement
            U.iloc[i] = diff;
            D.iloc[i] = 0;
        else:                   #   downward movement
            D.iloc[i] = abs(diff);
            U.iloc[i] = 0;

    for i in range(n, len(price.index)):
        avg_U = float(U.iloc[i-n:i+1].mean());  #   add +1 because does not consider last element
        avg_D = float(D.iloc[i-n:i+1].mean());
        
        if avg_U == 0 or avg_D == 0:
            date = RS.index[i]
            RS.iloc[i] = RS.iloc[i-1];          #   copy RS from previous day if cannot calculate
            #   print("Error: Divide by zero on date ", date, " U: ", avg_U, " D: ", avg_D);
        else:
            RS.iloc[i] = avg_U/avg_D;
            RSI.iloc[i] = 100-(100/(RS.iloc[i]+1));
            
    RSI.columns = (['RSI']);

    return RSI, RS
        
        


if __name__ == '__main__':

    start_date = '2012-11-26'       #   start 5 years ago
    end_date = '2017-11-10'
    syms = ['TSLA']                 #   Tesla
    rolling_window = 50             #   number of days per rolling window
    rsi_window = 14                #   number of days to use for RSI calculation
    risk_free_rate = 0.01           #   risk free rate of return for 1 year is 1%

    
    calculate_tech_feats(start_date, end_date, syms, risk_free_rate, rsi_window, rolling_window);
    
    
    
