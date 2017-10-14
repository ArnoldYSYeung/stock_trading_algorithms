##  Author: Arnold Yeung (arnold.yeung@alumni.ubc.ca)
##  Date:   October 5th, 2017
##  Uses Random Forest to predict the return for MSCI Emerging Markets (EM)
##  index based on other index returns
##  Expected runtime is O(m * n * log(n)), where m = num of trees, n = num of samples
##
##  Reference: http://scikit-learn.org/stable/modules/ensemble.html#forest
##
##  2017-10-07: Implement basis for RF algorithm.  Currently, training and testing
##              on same dataset.  Need to implement cross-validation or test set
##  2017-10-14: Separate training and testing samples.  Current R^2 is 0.68 - optimization
##             may be necessary.  Add in plotting function.



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def learn_rt(returns_file, leaf_size = 1, train_pct = 60, verbose = False):

    #   extract index return data from .csv
    returns = pd.read_csv("data/{}".format(returns_file), index_col = 'Date', \
                          parse_dates = True, na_values = ['nan'])

    y = returns.iloc[:,-1]      #   target values
    X = returns.iloc[:,:-1]     #   features

    n_test = round(len(X)*(100-train_pct)/100)  #   number of test samples
    n_train = len(X) - n_test   #   number of training samples
    n_features = len(X.columns) #   number of features
    
    #   extract top train_pct of data to use as training value
    #   NOTE: earlier data is taken first, to minimize influences of past
    #           events on future events
    X_train = returns.iloc[1:n_train, :-1]
    y_train = returns.iloc[1:n_train, -1]

    X_test = returns.iloc[n_train:, :-1]
    y_test = returns.iloc[n_train:, -1]

    print("Training Random Forest Regressor model...")
    clf = RandomForestRegressor(n_estimators = 300, max_features = n_features, \
                                n_jobs = -1, oob_score = True)
    #   n_estimator = num of trees, more the better (but higher computation)
    #   max_features = size of subset of features to randomly select (1 feature
    #                   may be selected multiple times in 1 estimator)
    #   n_jobs = number of allowed parallelization cores (-1 = as many as possible)
    
    clf = clf.fit(X_train,y_train)
    #   Built with bootstrap sampling (e.g. with replacement)
    #   Random subset of features selected to build each tree
    #   Split is chosen as best split among random subset of features (increases bias,
    #   but variance decreases due to averaging)

    #   predict with test set
    return_predict = clf.predict(X_test).reshape((n_test,1))
    score = clf.score(X_test, y_test)     #   coefficient of determination R^2 of prediction
    #   Training set should be different from test set.
    
    print("Coefficient of Determination: ", score);

    prediction = pd.DataFrame(0, index = X_test.index, \
                              columns = ['Actual', 'Predicted'])

    prediction[['Actual','Predicted']] = [y_test.values.reshape(n_test, 1), \
                                          return_predict]
    print(prediction)

    #   plot Predicted v. Actual
    plt.scatter(prediction['Predicted'], prediction['Actual'])
    plt.title("Random Forest Regressor")
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    
    plt.show()
    
    
if __name__ == '__main__':

    returns_file = 'istanbul.csv'
    train_pct = 60;             #   percentage of data to use for training
    learn_rt(returns_file, train_pct)
