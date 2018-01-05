import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def extract_feature_per_person(person):
    """
    For each person in the dataset, smooth data, fill missing values and perform dimension reduction.
    Use ADL1~3 for training, ADL4~5 for testing.
    Return:
    train_reduced: [n_train, n_features]. Sequence of training feature vectors.
    test_reduced: [n_test, n_features]. Sequence of testing feature vectors.
    train_labels: [n_train, ]. Sequence of training activity labels in {0, 101, 102, 103, 104, 105}
    test_labels: [n_test, ]. Sequence of testing activity labels in {0, 101, 102, 103, 104, 105}
    train_len: length of subsequences in training data. sum(train_len) = n_train.
    test_len: length of subsequences in testing data. sum(test_len) = n_test.
    """
    # Load data
    sadl_n = []
    for n in range(1, 6):
        sadl_n.append(pd.read_table('data/S%d-ADL%d.dat' % (person, n), sep='\s+', header=None, dtype=float))

    # Smooth data, time: col 0, features: col 1~36, labels: col 244 
    winsize = 15
    stepsize = 8
    # train data
    train_sample = np.empty((0, 36))
    train_labels = np.empty((0))
    train_len = []
    for i in range(0, 3):
        features = moving_avg(sadl_n[i].iloc[:, 1:37], winsize, stepsize)
        labels = moving_vote_majority(sadl_n[i].iloc[:, 244], winsize, stepsize)
        train_sample = np.concatenate((train_sample, features), axis=0)
        train_len.append(features.shape[0])
        train_labels = np.concatenate( (train_labels, labels) )
    train_len = np.array(train_len)
    # test data
    test_sample = np.empty((0, 36))
    test_labels = np.empty((0))
    test_len = []
    for i in range(3, 5):
        features = moving_avg(sadl_n[i].iloc[:, 1:37], winsize, stepsize)
        labels = moving_vote_majority(sadl_n[i].iloc[:, 244], winsize, stepsize)
        test_sample = np.concatenate((test_sample, features), axis=0)
        test_len.append(features.shape[0])
        test_labels = np.concatenate( (test_labels, labels) )
    test_len = np.array(test_len)

    # Fill missing values
    col_threshold = 0.5
    train, test = fill_missing(train_sample, test_sample, col_threshold, True)

    # Normalize features
    scalar = StandardScaler() # center to mean and normalize to unit variance
    train_normalized = scalar.fit_transform(train)
    test_normalized = scalar.fit_transform(test)

    # Dimension reduction
    pca = PCA()
    pca.fit(train_normalized)
    var_thres = 0.95 # keep components to up to 95% total variance
    n_comp = (pca.explained_variance_ratio_.cumsum() < var_thres).sum() + 1

    pca_train = PCA(n_components=n_comp)
    train_reduced = pca_train.fit_transform(train_normalized)
    test_reduced = pca_train.transform(test_normalized)

    return train_reduced, test_reduced, train_labels, test_labels, train_len, test_len
    
