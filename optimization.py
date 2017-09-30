##  Author: Arnold Yeung
##  Date:   September 19th, 2017
##  This script and these functions receive in options for a stock portfolio and
##  calculates the distribution of funds to optimize the indicated statistics of
##  the portfolio.
##
##  2017-09-19: prop = 'adr' option does not seem to be optimizing - reason unknown
##  2017-09-30: remove prop = 'adr' option because output should correspond with
##              prop = 'cr' theoretically

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import assessment as ass

 

def optimize_portfolio(sd, ed, syms, rfr, sf, start_val, gen_plot, prop):
#   Optimizes the portfolio to yield the maximum property

    #   create price dataframe for syms
    dates = pd.date_range(sd, ed);          #   create date range
    prices = ass.create_price_df(syms, dates);            #   create price df

    #   find optimal allocations for portfolio
    init_allocs = [1./len(prices.columns),]*len(prices.columns);    #   divide init allocs equally
    allocs = find_optim_allocs(prices, init_allocs, rfr, sf, start_val, prop);

    #   create portfolio based on allocs
    portfolio, total_vals = ass.create_portfolio(prices, allocs, start_val);

    #   calculate portfolio statistics
    cr = ass.get_cum_return(total_vals);
    dr, adr, sddr = ass.get_daily_returns(total_vals);
    sr = ass.get_sharpe_ratio(dr, rfr, sf);

    #   plot portfolio relative to benchmark ['SPY']
    if gen_plot == 'y':
        norm_benchmark = ass.norm_df(prices['SPY'], 0)*start_val;
        df_plot = pd.concat([total_vals, norm_benchmark], keys=['Portfolio', 'SPY'], axis=1);
        ass.plot_data(df_plot,'Portfolio Performance');
    

    return allocs, cr, adr, sddr, sr;
    #   returns cumulative return, average daily return, std of daily return,
    #   sharpe ratio

def find_optim_allocs(price_df, init_allocs, rfr, sf, start_val, prop):
#   Find optimal distribution of funds to portfolio stocks to optimize based on
#   indicated statistical property

    #   all allocs elements must be between 0 and 1
    
    bnds = tuple((0, 1) for x in range (len(init_allocs))); #   four (0, 1) in a tuple
    cnst = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #   sum - 1 = 0

    if prop == 'cr':
        print('Optimizing for Cumulative Return...');
        opts = spo.minimize(min_cr, init_allocs, args = (price_df,start_val),
                            method = 'SLSQP', bounds = bnds, constraints = cnst);
##    elif prop == 'adr':
##        #   Testing shows that adr is not optimizing but staying at init_allocs
##        print('Optimizing for Average Daily Return...');
##        opts = spo.minimize(min_adr, init_allocs, args = (price_df,start_val),
##                            method = 'SLSQP', bounds = bnds, constraints = cnst);
    elif prop == 'vol':
        print('Optimizing for Volatility...');
        opts = spo.minimize(min_vol, init_allocs, args = (price_df,start_val),
                            method = 'SLSQP', bounds = bnds, constraints = cnst);        
    else:
        print('Optimizing for Sharpe Ratio...');
        opts = spo.minimize(min_sharpe, init_allocs, args = (price_df,start_val,rfr, sf),
                            method = 'SLSQP', bounds = bnds, constraints = cnst);
    

    allocs = opts['x']; #   allocs stored in 'x' variable in opts

    return allocs;
    
def min_sharpe(allocs, prices, start_val, rfr, sf):
    portfolio, total_vals = ass.create_portfolio(prices, allocs, start_val); #   create portfolio
    dr, dr_mean, dr_std = \
                   ass.get_daily_returns(total_vals);   #   get daily return
    sr = ass.get_sharpe_ratio(dr, rfr, sf);
    
    return -1*sr;

def min_cr(allocs, prices, start_val):
    portfolio, total_vals = ass.create_portfolio(prices, allocs, start_val); #   create portfolio
    cr = ass.get_cum_return(total_vals);
    
    return -1*cr;

def min_adr(allocs, prices, start_val):
    #   no longer in use
    portfolio, total_vals = ass.create_portfolio(prices, allocs, start_val); #   create portfolio
    dr, dr_mean, dr_std = ass.get_daily_returns(total_vals);   #   get daily return
    return -1*dr_mean;

def min_vol(allocs, prices, start_val):
    portfolio, total_vals = ass.create_portfolio(prices, allocs, start_val); #   create portfolio
    dr, dr_mean, dr_std = \
                   ass.get_daily_returns(total_vals);   #   get daily return
    return dr_std;


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
    syms = ['SPY', 'FB'];
    rfr = 0.01;
    sf = 252;           #   risk free rate of return for 1 year is 1%
    gen_plot = 'y';     #   'y' or 'n'
    prop = 'sharpe';       

    allocs, cr, adr, sddr, sr = optimize_portfolio(start_date, end_date, syms,
                                                  rfr, sf, start_val, gen_plot, prop)

    print('Stock Allocations: ', allocs);
    print('Cumulative Return: ', cr);
    print('Average Daily Return: ', adr);
    print('Volatility (StD Daily Return: ', sddr);
    print('Sharpe Ratio: ', sr);
