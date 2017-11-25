##  Author: Arnold Yeung
##  Date:   September 18th, 2017
##  This script and these functions assess the value of a portfolio by outputting
##  key portfolio statistics: cumulative return, avg daily return, Sharpe ratio,
##  volatility (as standard deviation)
##
##  2017-11-25: fixed bug in create_price_df() where orig_syms is undefined because ref is in
##              syms
##  2017-09-30: add in ref='SPY' parameter for create_price_df 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def assess_portfolio(sd, ed, syms, allocs, sv, rfr, sf, gen_plot):
#   assesses the past performance of a portfolio
#   Input: start date, end date, symbols of portfolio, allocation of
#           symbols, start value of portfolio, risk free return (constant),
#           sampling frequency per year, plot option
#   Output: cumulative return, average daily return,, standard deviation
#           of daily return, Sharpe Ratio

    dates = pd.date_range(sd, ed);          #   create date range
    price = create_price_df(syms, dates);            #   create price df
    portfolio, total_vals = create_portfolio(price, allocs, sv);    #   create portfolio df and total value
    cr = get_cum_return(total_vals)
    dr, adr, sddr = get_daily_returns(total_vals);
    sr = get_sharpe_ratio(dr, rfr, sf);

    if gen_plot == 'sum':
        plot_data(total_vals, 'Summed Portfolio Performance');
    elif gen_plot == 'ind':
        plot_data(portfolio, 'Individual Portfolio Stock Performance');
    else:
        print('No plot');
    
    return cr, adr, sddr, sr
    #   cumulative return, average daily return,
    #   standard deviation of daily return, Sharpe Ratio

def create_price_df(syms, dates, ref = 'SPY'):
#   returns dataframe containing data found in location str (.csv)
#   uses SPY as reference as default (ref)
    df = pd.DataFrame(index=dates);     #   create empty dataframe with dates as index

    orig_syms = syms            # does not contain ref if not already in syms

    if ref not in syms:
        syms = [ref] + syms        #   add reference into symbols (used for determining trading days)

    for symbol in syms:      #   for each symbol
        #   create temp df for each syms
        df_syms = pd.read_csv("data/{}.csv".format(symbol), index_col = 'Date',
                                                   parse_dates = True, usecols =["Date", "Adj Close"],
                                                   na_values = ['nan'])
        df_syms = df_syms.rename(columns = {'Adj Close': symbol});

        df = df.join(df_syms);     # concatenate df_syms to main df

        if symbol == ref:
            df = df.dropna(subset=[ref])      # remove dates where SPY did not trade

    df.fillna(method='ffill', inplace = True);    # fill missing data based on last previous value (if available)
    df.fillna(method='bfill', inplace = True);     # fill remaining missing data based on first future value

    if ref not in orig_syms:
        df = df.drop(ref,1)        #   remove ref column from df

    return df

def norm_df(df, index):
#   normalizes the price dataframe based on row index
    df = df/df.ix[index];

    return df

def create_portfolio(price, allocs, start_val):
#   create portfolio of stock values and returns dataframe of individual
#   stock values and dataframe of total values
    price = norm_df(price, 0);                  #   normalize to row 0
    portfolio = price * allocs * start_val;    #   get value of portfolio   
    total_vals = portfolio.sum(axis = 1);               #   sum per row

    return portfolio, total_vals
    
def get_cum_return(total_vals):
    cum_return = total_vals[-1]/total_vals[0]-1;    #   gain percentage of last value from first value
    return cum_return

def get_daily_returns(total_vals):
#   return portfolio daily returns, the average portfolio daily return,
#   and the std of the portfolio daily returns 
    daily_returns = total_vals.copy();                              #   copy same size dataframe
    daily_returns[1:] = total_vals[1:]/total_vals[:-1].values-1;    #   compute daily returns starting row 1
    # values is necessary to prevent pandas from doing an element-wise operation
    # i.e. dividing 2 matrices together

    daily_returns.ix[0] = 0;          # set first row as 0 since no return on first day

    return daily_returns, daily_returns.mean(), daily_returns.std()

def get_sharpe_ratio(daily_returns, risk_free_rate, sampling_freq):
#   calculate sharpe ratio from daily_returns
    daily_rfr = get_daily_risk_free(risk_free_rate, sampling_freq);     # calculate daily risk free rate
    risk_rate = daily_returns-daily_rfr;
    sharpe = risk_rate.mean()/risk_rate.std();
    
    return sharpe

def get_daily_risk_free(risk_free_rate, sampling_freq):
#   calculate the daily risk free rate based on an inputted risk free rate (e.g. per annum)
#   and a sampling frequency representing the number of times the daily risk free rate goes
#   into the inputted sampling rate (e.g. for a year, 252 days = 252)
    return (1 + risk_free_rate)**(1./sampling_freq) - 1

def plot_data(df, title):
    df_plot = df.plot(title = title, fontsize = 12);
    df_plot.set_xlabel("Date");
    df_plot.set_ylabel("Value");
    plt.show()
    return 0

if __name__ == '__main__':    

    start_date = '2015-11-23';
    end_date = '2017-03-27';
    start_val = 100;
    dates = pd.date_range(start_date, end_date);
    syms = ['SPY', 'FB', 'GLD', 'IBM'];
    allocs = [0.35, 0.30, 0.2, 0.15];
    rfr = 0.01;
    sf = 252;           #   risk free rate of return for 1 year is 1%
    gen_plot = 'ind';   #   'sum' or 'ind'
    
    cr, adr, sddr, sr = assess_portfolio(start_date, end_date, syms,
                                         allocs, start_val, rfr, sf, gen_plot);

    print("Cumulative Return: ", cr)
    print("Average Daily Return: ", adr)
    print("Daily Return StD: ", sddr)
    print("Sharpe Ratio: ", sr)

