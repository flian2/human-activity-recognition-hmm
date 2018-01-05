import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from clean_data import extract_feature_per_person
from hmm import DiscreteDistr, GaussDistr, GaussMixDistr
from hmm import MarkovChain
from hmm import HMM, make_leftright_hmm
from hmm_twolayer_model import *


def search_nmix_gmm_crossval(train, train_labels, train_len, n_states):
    """
    Greedily search the the number of mixtures for each label, n_mix from 1 to 6 over all labels.
    Use k-fold cross validation for highest mean F1 score, where k is the number of training subsequences.
    Return:
    n_mix_opt: The chosen number of mixtures for each label.
    """
    start_ind = np.append(np.array([0]), np.cumsum(train_len)[:-1].astype(int))
    n_mix = np.ones((n_states), dtype=int)
    max_score = 0 # keep track of max F1 score
    for label in range(0, 6):
        n_opt = 1
        # search n_mixture from 1 to 10, keep the n_mixture for other labels fixed. 
        for nm in range(1, 11):
            n_mix[label] = nm
            f1_s = []
            for fold in range(0, len(train_len)):
                # split train and validation set
                val_mask = np.zeros((train.shape[0])).astype(bool)
                val_mask[start_ind[fold]: start_ind[fold] + train_len[fold]] = True
                train_mask = np.logical_not(val_mask)
                train_sub = train[train_mask, :]
                train_sub_labels = train_labels[train_mask]
                train_sub_len = train_len[np.arange(0, len(train_len)) != fold] 

                val_sub = train[val_mask, :]
                val_sub_labels = train_labels[val_mask]
                val_sub_len = [train_len[fold]]
                # cross validate
                mc_top, prob_mass_top, sub_mcs, gmms = \
                    hmm_twolayer_train(train_sub, train_sub_labels, train_sub_len, n_mix)
                val_states, predicted_labels = \
                    hmm_twolayer_predict(val_sub, val_sub_len, mc_top, prob_mass_top, sub_mcs, gmms)

                true_labels = np.maximum(val_sub_labels - 100, 0)
                f1_s.append(f1_score(true_labels, predicted_labels, average='weighted'))
            if np.mean(f1_s) > max_score:
                max_score = np.mean(f1_s)
                n_opt = nm
            print "current n_mix: ", n_mix
            print "currrent max f1 score ", max_score

        n_mix[label] = n_opt
    return n_mix, max_score


def main():
    person = int(sys.argv[1])
    # Feature extraction
    train_reduced, test_reduced, train_labels, test_labels, train_len, test_len = \
        extract_feature_per_person(person)
    n_mix = [6, 6, 6, 2, 2, 6]
    f1 = hmm_gmm_F1score(train_reduced, test_reduced, train_labels, test_labels, train_len, test_len, n_mix)
    print f1
    # n_mix, max_f1 = search_nmix_gmm_crossval(train_reduced, train_labels, train_len, 6)
    # print n_mix
    # print max_f1


if __name__ == '__main__':
    main()