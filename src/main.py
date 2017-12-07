from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import dataloader as dtl
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from pymf import pca, cmeans, aa


def threshold_features(X, float_to_int_idxs, float_to_binary_idxs):
    """
    Sanitize the matrix X by rounding floats to ints and thresholding floats
    to binary variables at the provided indices of X's feature vectors.
    Assumes X is of the format samples X features
    """
    num_samples = X.shape[0]
    for index in float_to_int_idxs:
        for sample in range(num_samples):
            X[sample, index] = int(round(X[sample, index]))

    for index in float_to_binary_idxs:
        for sample in range(num_samples):
            cur_val = X[sample, index]
            if cur_val >= 0.5:
                X[sample, index] = 1
            else:
                X[sample, index] = 0

if __name__ == '__main__':

    #Feature indices
    TEMPERATURE_INDX = 7
    AGE_INDX = 13
    SEX_INDX = 14
    NERVE_TYPE_INDX = 35
    HYPERPOL_IV_INDX = 24
    REFRACTORINESS_2_MS = 28
    REFRACTORINESS_2_POINT_5_MS = 25
    TEh_OVERSHOOT = 20

    #Parameters
    epsilon = 1.0e-6
    train_size = 125
    test_size = 40
    num_factors = 3
    max_factors = 36
    algs = ['pca', 'cmeans', 'aa']
    num_algs = len(algs)
    alg_errors = [[] for alg in range(num_algs)]
    alg_error_mean = [[] for alg in range(num_algs)]
    alg_error_std = [[] for alg in range(num_algs)]

    #NOTE: REFRACTORINESS_2_POINT_5_MS must come after REFRACTORINESS_2_MS in the missing_feature_indxs list
    missing_feature_indxs = [HYPERPOL_IV_INDX, REFRACTORINESS_2_MS, REFRACTORINESS_2_POINT_5_MS , TEh_OVERSHOOT] #Features from which data is randomly deleted in the test set
    percent_missing_data = [0.175, 0.30, 0.0625, 0.047]

    #To ensure re-running the experiment will yield the same results
    np.random.seed(0)

    #Indexes into the feature vector that need to be thresholded to valid values
    #After the matrix is reconstructed
    float_to_int_idxs = [TEMPERATURE_INDX, AGE_INDX]
    float_to_binary_idxs = [SEX_INDX, NERVE_TYPE_INDX]

    train_set, test_set = dtl.load_nerve_data(train_size, test_size)

    #NOTE: #In pymf the matrix X is factorized such that X = W * H, where
    #W : "m x k" matrix of basis vectors
    #H : "k x n" matrix of coefficients

    #Thus, in comparison to our notes, where X = P * F
    #W = transpose(F)
    #H = transpose(P)

    #Thus, W * H is an m X k * k X n = m X n result, instead of the expected n X k
    #Therefore, we need to transpose our data matrix, prior to factorizing, then
    #transpose the output back to use the formulas as written in our initial draft
    train_set = np.transpose(train_set)

    #Remove some of the data from a copy of the test set
    test_set_missing = np.array(list(test_set))
    for (feature_indx, missing_data_percent) in zip(missing_feature_indxs, percent_missing_data):

        #We need to enforce the constraint that missing refractory feature data at 2.5MS is a subset of the 2MS samples,
        if feature_indx != REFRACTORINESS_2_POINT_5_MS:
            samples_to_modify_indxs = np.random.choice(test_size, int(round(test_size * missing_data_percent)), replace=False)
        else:
            samples_to_modify_indxs = np.random.choice(refractory_2_ms_samples, int(round(test_size * missing_data_percent)), replace=False)

        #We save the indices used for deleting refractory feature data at 2MS
        if feature_indx == REFRACTORINESS_2_MS:
            refractory_2_ms_samples = samples_to_modify_indxs
        test_set_missing[samples_to_modify_indxs, feature_indx] = 0

    #Ensure that pca is behaving as expected: that is, that the factorization error using all k factors is minimal
    cur_mdl = pca.PCA(train_set, num_bases=max_factors)
    cur_mdl.factorize()
    assert cur_mdl.ferr < epsilon

    #Ensure that the right features are getting thresholded to the right values
    dummy_X_hat = np.dot(cur_mdl.W, cur_mdl.H)
    threshold_features(dummy_X_hat, float_to_int_idxs, float_to_binary_idxs)
    assert all(map(lambda x: x % 1 == 0, dummy_X_hat[:, 7]))
    assert all(map(lambda x: x % 1 == 0, dummy_X_hat[:, 13]))
    assert all(dummy_X_hat[:, 14]) in [0, 1]
    assert all(dummy_X_hat[:, 35]) in [0, 1]

    #Run the main experiment
    for alg in range(num_algs):
        cur_alg = algs[alg]
        print(('Running on train={0} and test={1} samples for algorithm: {2}, with {3} factors').format(train_set.shape[1], test_set.shape[0], cur_alg, num_factors))

        if cur_alg == 'pca':
            cur_mdl = pca.PCA(train_set, num_bases=num_factors)
        elif cur_alg == 'cmeans':
            cur_mdl = cmeans.Cmeans(train_set, num_bases=num_factors)
        elif cur_alg == 'aa':
            cur_mdl = aa.AA(train_set, num_bases=num_factors)
        else:
            exit("ERROR: Invalid algorithm {0} selected!!!".format(cur_alg))
        cur_mdl.factorize()

        F = np.transpose(cur_mdl.W)
        X_hat = np.dot(np.dot(test_set, np.linalg.pinv(F)), F)
        threshold_features(X_hat, float_to_int_idxs, float_to_binary_idxs)

        #Compute the current algorithm error across all of the samples
        cur_alg_errors = []
        for i in range(test_size):
            cur_alg_errors.append(np.linalg.norm(X_hat[i, :] - test_set[i, :], ord=2))
        alg_errors[alg] = cur_alg_errors
        alg_error_mean[alg] = np.mean(np.array(cur_alg_errors))
        alg_error_std[alg] = np.std(np.array(cur_alg_errors))

    print("Displaying the results for all algorithms...")
    for alg in range(num_algs):
        print("Alg: {0} Error: {1} Standard Deviation: {2}").format(algs[alg], alg_error_mean[alg], alg_error_std[alg])
        print("Error Values: " + str(alg_errors[alg]))
        print("\n")

    #Set up the boxplot
    x_vals = [[] for alg in range(num_algs)]
    for i in range(num_algs):
        x_vals[i] = np.random.normal(i+1, 0.04, test_size)
    plt.boxplot(alg_errors, labels=algs)
    plt.xlabel("Algorithm")
    plt.ylabel("Error (per data instance)")

    #Plot the scatter plot for each group, with a different colour for each
    clevels = np.linspace(0., 1., num_algs)
    for x, y, clevel in zip(x_vals, alg_errors, clevels):
        plt.scatter(x, y, c=mpl.cm.prism(clevel), alpha=0.4)
    plt.show()
