import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from clean_data import extract_feature_per_person
from hmm import DiscreteDistr, GaussDistr, GaussMixDistr
from hmm import MarkovChain
from hmm import HMM, make_leftright_hmm

def make_outputdistr_given_nmix(train, train_labels, class2label, n_mix):
    # given the number of components for each label
    outputdistr_stats = pd.DataFrame(index=class2label, columns=['n_mixture', 'mean_loglikelihood'])
    outputdistr_gmm = []
    
    for i, label in enumerate(class2label):
        gmm = GaussMixDistr(gauss=n_mix[i])
        gmm.init_by_data(train[train_labels == label, :])
        gmm.train(train[train_labels == label, :])
        outputdistr_gmm.append(gmm)
        outputdistr_stats.iloc[i, :] = [n_mix[i], gmm.logprob(train[train_labels == label, :]).mean()]
    return outputdistr_gmm, outputdistr_stats

def loglikelihood(hmm_top, x, gmms):
    """
    Compute log likelihood for each observation sample given the MarkovChain of top-layer hmm, 
    and gmm distribution P(X(t) | activity label = i).
    Input:
    hmm_top: top-layer hmm
    gmms: list of gmm object. gmms[i] has P(X(t) | activity_label = i)
    Return:
    logp_x: [n_states, n_samples]. logp_x[i, t] = log P[X(t) | S_{hmm_top} = i]
    Method:
    Compute: P(X(t) | state = i ) for t = 0...T, i = 0...5,
    P(X(t) | state = i ) = \sum_{label i} P(X(t) | label = j, state = i) * P(label = j | state = i)
    """
    T = x.shape[0]
    n_states = hmm_top.n_states
    logp_x = np.zeros((n_states, T))
    for state in range(0, hmm_top.n_states):
        p0 = hmm_top.output_distr[state].prob_mass
        label_ind = np.argwhere(p0 > 0)
        p0 = p0[p0 > 0]
        logprob_per_label = gmms[0].logprob(x, [gmms[i] for i in label_ind])
        logp_x[state, :] = np.log( p0[np.newaxis, :].dot(np.exp(logprob_per_label)) )[0, :]

    return logp_x

def viterbi_state_sequence(hmm_top, x, x_len, gmms):
    """
    Predict top-layer hmm state sequence using viterbi algorithm.
    Input
    hmm_top: top-layer hmm
             hmm_top.state_gen: MarkovChain for top-layer hmm.
             hmm_top.output_distr: hmm_top.output_distr[i][j] = P[activity_label = j | S_{hmm_top} = i]
    x: [T, data_size]. observation vector sequence stacked together
    x_len: length of subsequences
    gmms:      list of gmm object. gmms[i] has P(X(t) | activity_label = i)
    Return:
    s_opt: [T, ] predicted top-layer hmm state sequence
    logP: [n_seq, ] logP of each subsequence
    """
    start_ind = 0
    s_opt = np.zeros((x.shape[0]))
    logP = np.zeros((len(x_len)))
    for i in range(0, len(x_len)):
        logp_x = loglikelihood(hmm_top, x[start_ind:start_ind + x_len[i], :], gmms)
        s_opt[start_ind: start_ind + x_len[i]], logP[i] = hmm_top.state_gen.viterbi(logp_x)
        start_ind += x_len[i]
    return s_opt - 1, logP # the state sequence index from 0

def make_sub_hmm_mc(train_states, train_len, train_labels, n_states):
    """
    Initialize and train the sub-layer hmm for each top-layer hmm state.
    For the sub-layer hmm, each state corresponds to one activity label. 
    We train the transition prob by running Baum-Weltch EM training, by initializing the output 
    probability mass as the diagonal matrix, to force one to one mapping between sub-layer state and label.
    Input:
    train_states: [n_samples, ]. Sequence of top-layer hmm states.
    train_len: list. Length of subsequences.
    train_labels: [n_samples, ]. Activity labels of training sequence. The range of each label must be in [0, n_states)
    Return:
    mc_per_state: list of MarkovChain objects. mc_per_state[i] is the MarkovChain for top-layer state i. 
    """
    train_labels = train_labels[:, np.newaxis]
    start_ind = [0] + list(np.cumsum(train_len)[:-1].astype(int))
    mc_per_state = []
    for n in range(0, n_states):
        x_labels = train_labels[train_states == n, :]
        x_len = np.array([np.sum(train_states[s:s + train_len[i]] == n) for i, s in enumerate(start_ind)])
        x_len = x_len[x_len > 0]
        # Aii = 1 - 1/state_duration, Aij = 1/state_duration / (n_states_actual - 1)
        A0 = np.eye((n_states))
        D = np.array([np.sum(x_labels == i) / float(len(x_len)) for i in range(0, n_states)])
        n_states_actual = np.sum(D > 0)
        for i in range(0, n_states):
            if D[i] == 0:
                continue
            A0[i, D > 0] = 1.0 / D[i] / n_states_actual
            A0[i, i] = 1.0 - 1.0 / D[i]
        p0 = np.ones((n_states)) / float(n_states_actual)
        p0[D == 0] = 0.0
        prob_mass = np.eye((n_states))
        pD = [DiscreteDistr(prob_mass[i, :]) for i in range(0, n_states)]
        mc = MarkovChain(p0, A0)
        hmm_mc = HMM(mc, pD)
        hmm_mc.train(obs_data=x_labels + 1, l_data=x_len) # discrete distribution index from 1 
        mc_per_state.append(hmm_mc.state_gen)
    return mc_per_state

def predict_subhmm_labels(states_tophmm, obs_data, l_data, sub_mcs, outputdistr):
    """
    Predict activity labels inside each top-layer hmm state.
    Input:
    states_tophmm: [n_samples, ]. Sequences of top-layer hmm states.
    obs_data: [n_samples, n_features]
    l_data: length of sequences. sum(l_data) = len(states_tophmm)
    sub_mcs: MarkovChain objects for each top-layer state. The order of states in the mc corresponds to the sub-level label.
    outputdistr: the output distribution for the states in sub_mcs.
    Return:
    labels_opt: [n_samples, ]. predicted labels, range from 0 to n_states - 1.
    """
    start_ind = [0] + list(np.cumsum(l_data)[:-1].astype(int))
    labels_opt = np.zeros((states_tophmm.shape[0]))
    cur_pos = 0
    for t in range(0, len(l_data)):
        state_subseq = states_tophmm[start_ind[t]:start_ind[t] + l_data[t]]
        obs_subseq = obs_data[start_ind[t]:start_ind[t] + l_data[t], :]
        diff = np.append(np.array([1]), state_subseq[1:] - state_subseq[:-1])
        i_newstate = np.append( np.argwhere(diff != 0), np.array([len(diff)]) )
        for m in range(0, len(i_newstate) - 1):
            state_tt = int(state_subseq[i_newstate[m]])
            obs_tt = obs_subseq[i_newstate[m]:i_newstate[m + 1], :]
            # run viterbi on markov chain sub_mcs[state_tt]
            logp_x = outputdistr[0].logprob(obs_tt, outputdistr)
            labels_tt, logP = sub_mcs[state_tt].viterbi(logp_x)
            labels_opt[cur_pos:cur_pos + len(labels_tt)] = labels_tt - 1 # make the label index starting from 0
            cur_pos += len(labels_tt)
    return labels_opt

def hmm_twolayer_train(train, train_labels, train_len, n_mix):
    """
    Train a twolayer hmm model.
    Return:
    hmm_top: top-layer hmm object.
    sub_mcs: list of MarkovChain objects for sub-layer hmm.
    gmms: GMM output distribution for sub-layer hmm.
    """
    # Train gmm model given the number of components in each label
    gmms, stats = make_outputdistr_given_nmix(train, train_labels, [0, 101, 102, 103, 104, 105], n_mix)

    # Train top-layer left-right hmm with discrete output distribution. Output distribution: P(activity label | hmm state)
    label_transfer = (np.maximum(train_labels - 100, 0) + 1)[:, np.newaxis]
    # Transform the labels into range 1~6. {0: 1, 101: 2, 102: 3, 103: 4, 104: 5, 105: 6}
    discreteD = DiscreteDistr(np.ones((6))) # a discrete distribution with 6 possible output
    n_states = 6
    hmm_top = make_leftright_hmm(n_states, discreteD, obs_data=label_transfer, l_data=train_len)
    for i in range(0, n_states):
        hmm_top.output_distr[i].prob_mass[hmm_top.output_distr[i].prob_mass < 1e-2] = 0
    
    # Train a sub-layer ergodic hmm for each top-layer hmm state, 
    # the training state sequence for hmm_top can be predicted using viterbi algorithm.
    # For the bottom-layer hmm:
    # state: activity label {e.g. state 0: label 0, state 1: label 101}
    # output distribution: gmm 
    
    # Recover state sequnce of top-layer hmm
    train_states, _ = hmm_top.viterbi(label_transfer)
    train_states -= 1
    # train_states, _ = viterbi_state_sequence(hmm_top.state_gen, train, train_len, prob_mass, gmms)
    # Sub-layer hmm training
    sub_mcs = make_sub_hmm_mc(train_states, train_len, np.maximum(0, train_labels - 100), n_states)

    return hmm_top, sub_mcs, gmms

def hmm_twolayer_predict(hmm_top, sub_mcs, gmms, val, val_len):
    # Predict validation set labels
    val_states, _ = viterbi_state_sequence(hmm_top, val, val_len, gmms)
    predicted_labels = predict_subhmm_labels(val_states, val, val_len, sub_mcs, gmms)
    return val_states, predicted_labels

def hmm_gmm_F1score(train, val, train_labels, val_labels, train_len, val_len, n_mix):
    hmm_top, sub_mcs, gmms = hmm_twolayer_train(train, train_labels, train_len, n_mix)
    val_states, predicted_labels = hmm_twolayer_predict(hmm_top, sub_mcs, gmms, val, val_len)
    
    # Evaluate the weighted F1 score
    true_labels = np.maximum(0, val_labels - 100)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return f1
