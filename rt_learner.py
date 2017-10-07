##  Author: Arnold Yeung (arnold.yeung@alumni.ubc.ca)
##  Date:   October 5th, 2017
##  Uses Random Forest to predict the return for MSCI Emerging Markets (EM)
##  index based on other index returns
##  Expected runtime is O(m * n * log(n)), where m = num of trees, n = num of samples
##
##  2017-10-07: Implement basis for RF algorithm.  Currently, training and testing
##              on same dataset.  Need to implement cross-validation or test set
##  Reference: http://scikit-learn.org/stable/modules/ensemble.html#forest


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def learn_rt(returns_file, leaf_size = 1, verbose = False):

    #   extract index return data from .csv
    returns = pd.read_csv("data/{}".format(returns_file), index_col = 'Date', \
                          parse_dates = True, na_values = ['nan'])

    y = returns.iloc[:,-1]      #   target values
    X = returns.iloc[:,:-1]     #   features

    n_features = len(X.columns) #   number of features
    n_samples = len(X)

    print("Training Random Forest Regressor model...")
    clf = RandomForestRegressor(n_estimators = 300, max_features = n_features, \
                                n_jobs = -1, oob_score = True)
    #   n_estimator = num of trees, more the better (but higher computation)
    #   max_features = size of subset of features to randomly select (1 feature
    #                   may be selected multiple times in 1 estimator)
    #   n_jobs = number of allowed parallelization cores (-1 = as many as possible)
    
    clf = clf.fit(X,y)
    #   Built with bootstrap sampling (e.g. with replacement)
    #   Random subset of features selected to build each tree
    #   Split is chosen as best split among random subset of features (increases bias,
    #   but variance decreases due to averaging)

    #   predict with test set
    return_predict = clf.predict(X).reshape((n_samples,1))
    score = clf.score(X, y)     #   coefficient of determination R^2 of prediction
    print("Coefficient of Determination: ", score);

    prediction = pd.DataFrame(0, index = returns.index, columns = ['Return'])
    prediction = prediction + return_predict
    print(prediction)
    
if __name__ == '__main__':

    returns_file = 'istanbul.csv'
    learn_rt(returns_file)
