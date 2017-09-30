##  Author: Arnold Yeung
##  Date:   September 30th, 2017
##  This application receives a list of order commands (.csv) and a starting cash value.  Simulation
##  of the order commands provides output of daily value of portfolio.
##
##  2017-10-01: implement commission and market impact costs (deducted from cash amount)
##  2017-10-01: implement in leverage checker to block trades above max_lev [unchecked]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import assessment as ass

def compute_portvals(orders_file, start_val, commission, market_impact, max_lev):
    #   returns the daily portfolio total value based on inputted order commands in orders_file
    #   and the starting cash value (start_val)

    #   extract order commands from .csv
    orders = pd.read_csv("orders/{}".format(orders_file), index_col = 'Date',
                                                   parse_dates = True, na_values = ['nan'])

    sd = orders.index.values[0]                          #   start date (first order)
    ed = orders.index.values[len(orders.index)-1]     #   end date (last order)
    syms = orders['Symbol'].tolist()                        #   list all symbols used
    syms = sorted(list(set(syms)))                  #   remove duplicate symbols and sort alphabetically

    #   create price dataframe
    dates = pd.date_range(sd,ed)
    prices = ass.create_price_df(syms, dates)            #  price df
    prices['Cash'] = 1                                   #  add Cash column into prices

    #   create trade df (correspond to order commands)
    #   contains day-to-day information on how assets are changed
    trades = pd.DataFrame(0, index = prices.index, columns = sorted(syms) + ['Cash'])      #   trade df (all zeros)

    for order in range(0, len(orders.index)):         #   for every order

        order_command = orders.iloc[order]      #   get command for the order
        sym = order_command['Symbol']       #   get order symbol (e.g. 'FB')
        action = order_command['Order']    #   get order action (BUY or SELL)
        amount = order_command['Shares']    #   get amount of shares

        #   copy order commands into trade
        if action == 'BUY':
            trades.loc[orders.index.values[order],sym] = trades.loc[orders.index.values[order],sym]+amount
            #   deduct purchase cost (including market impact) from Cash
            trades.loc[orders.index.values[order],'Cash'] = trades.loc[orders.index.values[order],'Cash'] \
                                                            - amount * (prices.loc[orders.index.values[order], \
                                                                                  sym] * (1+market_impact))
            #   deduct commission costs from Cash
            trades.loc[orders.index.values[order],'Cash'] = trades.loc[orders.index.values[order],'Cash'] \
                                                            - commission
        elif action == 'SELL':
            trades.loc[orders.index.values[order],sym] = trades.loc[orders.index.values[order],sym]-amount
            #   add sale earnings (including market impact) to Cash
            trades.loc[orders.index.values[order],'Cash'] = trades.loc[orders.index.values[order],'Cash'] \
                                                            + amount * (prices.loc[orders.index.values[order], \
                                                                                  sym] * (1-market_impact))
            #   deduct commission costs from Cash
            trades.loc[orders.index.values[order],'Cash'] = trades.loc[orders.index.values[order],'Cash'] \
                                                            - commission
        else:
            print('Undefined order action...')

    #   create holdings df
    #   contains day-to-day information on how much asset is held
    holdings = pd.DataFrame(0, index = prices.index, columns = sorted(syms)+['Cash'])    #   holdings df (all zeros)
    holdings.ix[0, 'Cash'] = start_val                                  #   give starting value to first day

    #   sum trading activities into holdings
    holdings.ix[0] = holdings.ix[0] + trades.ix[0]  #   for first day
    if calculate_leverage(holdings.ix[0], prices.ix[0]) > max_lev:
        print('Trade prohibited...')
        holdings.ix[0] = holdings.ix[0] - trades.ix[0]

    for trade_day in range(1, len(trades.index)):                           #   for every trading day after first
        holdings.ix[trade_day] = holdings.ix[trade_day-1] + trades.ix[trade_day]

        if calculate_leverage(holdings.ix[trade_day], prices.ix[trade_day]) > max_lev:
            print(holdings.index.values[trade_day])
            print('Trade prohibited...')
            holdings.ix[trade_day] = holdings.ix[trade_day-1] - trades.ix[trade_day]    #   remove trade

    #   create values df
    #   contains (monetary) value of all assets
    values = prices * holdings      #   element-wise multiplication

    port_vals = values.sum(axis = 1)    #   day-to-day total sum of asset values

    return port_vals    #   df containing portfolio value per day from start to end

def calculate_leverage(holding, price):
    value = holding * price
    leverage = sum(abs(value.drop('Cash',0)))/(sum(value.drop('Cash',0))+value.loc['Cash'])
    return leverage

if __name__ == '__main__':

    order_file = 'orders_v1.csv'        #   file containing order commands
    start_val = 1000000                 #   starting cash amount
    commission = 9.95                   #   commission paid per transaction
    market_impact = 0.05                #   market impact of trade towards the stock
                                        #   e.g. if buying, price increases by 0.5% before purchase
                                        #   e.g. if selling, price decreases by 0.5% before sale
                                        #   assume deducted from cash value and does not affect future prices
    max_lev = 1.5                         #   max leverage allowed before prohibition of trade
    
    print(compute_portvals('orders_v1.csv',start_val, commission, market_impact, max_lev))
