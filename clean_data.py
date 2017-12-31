import numpy as np
import pandas as pd
from scipy import stats

def moving_avg(df, n, step):
    """
    moving avg with step size
    (Did not use pandas.rolling since step size not supported.)
    Input:
    df: Input dataframe. First column (index 0) is time stamp.
    n: window length
    step: step size of the moving average
    Return:
    data: 2D array with features, has less number of rows than input df. 
    """
    T = df.shape[0]
    new_t = range(0, T, step)
    data = np.zeros((len(new_t), df.shape[1]))

    for i, t in enumerate(new_t):
        data[i, :] = np.nanmean(df.iloc[t: t + n, :].values, axis=0) # ignore nan when computing mean.
    # discard the last window
    return data[:-1, :]

def moving_vote_majority(label_df, n, step):
    """
    Take majority vote of labels in the moving window.
    Input:
    label_df: dataframe with one column denoting the label stream.
    n: window size.
    step: step size of the moving window
    Return:
    labels: 1D array with labels.
    """
    T = label_df.shape[0]
    new_t = range(0, T, step)
    labels = np.zeros((len(new_t)))
    for i, t in enumerate(new_t):
        labels[i] = stats.mode(label_df.iloc[t: t + n].values)[0][0]
    return labels[:-1]

def fill_missing(training, test, col_threshold, replace):
    """
    training: 2D array
    test: 2D array or None. If test is None, only process training data.
    col_threshold: if nan values in one column is greater than col_threshold, the column is ignored.
    replace: True: replace nan value in each row. False: delete the row if it contains nan.
    """
    # delete nan columns
    training = training[:, np.sum(np.isnan(training), axis=0) < col_threshold * training.shape[0]]
    if test is not None:
        test = test[:, np.sum(np.isnan(test), axis=0) < col_threshold * test.shape[0]]

    # process nans in each row
    if not replace:
        training = training[np.logical_not(np.any(np.isnan(training), axis=1)), :]
        if test is not None:
            test = test[np.logical_not(np.any(np.isnan(test), axis=1)), :]
    else:
        # replace with previous values
        for i in range(0, training.shape[0]):
            if i == 0:
                training[i, np.isnan(training[i, :])] = 0.0
            else:
                training[i, np.isnan(training[i, :])] = training[i - 1, np.isnan(training[i, :])]
        # remove the column with zero variance
        var_train = np.var(training, axis=0)
        logical_mask = var_train > 0
        training = training[:, logical_mask]
        
        if test is not None:
            test = test[:, logical_mask]
            mean_train = np.mean(training, axis=0)
            for i in range(0, test.shape[0]):
                if i == 0:
                    # fill in the mean of training data
                    test[i, np.isnan(test[i, :])] = mean_train[np.isnan(test[i, :])]
                else:
                    test[i, np.isnan(test[i, :])] = test[i - 1, np.isnan(test[i, :])]
    
    return training, test
