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


def sanitize_features(X, float_to_int_idxs, float_to_binary_idxs):
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
    epsilon = 1.0e-6
    train_size = 125
    test_size = 40
    num_factors = 3
    max_factors = 36
    percent_missing_data = 0.10 #Percentage of data deleted from the test set to simulate missing data
    graph_colours = ['r', 'g', 'b']
    algs = ['pca', 'cmeans', 'aa']
    num_algs = len(algs)
    alg_error = [[] for alg in range(num_algs)]
    alg_std = [[] for alg in range(num_algs)]
    #To ensure re-running the experiment will yield the same results
    np.random.seed(0)

    #Indexes into the feature vector that need to be thresholded to valid values
    #After the matrix is reconstructed
    #Corresponds to temperature and age, respectively
    float_to_int_idxs = [7, 13]
    #Corresponds to sex and nerve type (leg or arm), respectively
    float_to_binary_idxs = [14, 35]

    train_set, test_set = dtl.load_nerve_data(train_size, test_size)
    rand_test_missing_indices = np.random.choice(test_set.shape[0], int(round(test_size * percent_missing_data)), replace=False)

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

    #Remove some of the data from the test set
    test_set_missing = test_set[:]
    #TODO: How do we want to represent missing data, just set to 0?
    test_set_missing[rand_test_missing_indices, :] = 0

    #Ensure that pca is behaving as expected: that is, that the factorization error using all k factors is minimal
    cur_mdl = pca.PCA(train_set, num_bases=max_factors)
    cur_mdl.factorize()
    assert cur_mdl.ferr < epsilon

    #Ensure that the right features are getting thresholded to the right values
    dummy_X_hat = np.dot(cur_mdl.W, cur_mdl.H)
    sanitize_features(dummy_X_hat, float_to_int_idxs, float_to_binary_idxs)
    assert all(map(lambda x: x % 1 == 0, dummy_X_hat[0:5, 7]))
    assert all(map(lambda x: x % 1 == 0, dummy_X_hat[0:5, 13]))
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
        X_hat = np.dot(np.dot(test_set_missing, np.linalg.pinv(F)), F)
        #Threshold certain feature values  as appropriate after factorizing
        sanitize_features(X_hat, float_to_int_idxs, float_to_binary_idxs)

        #Compute the current algorithm error across all of the samples
        cur_alg_errors = []
        for i in range(test_size):
            cur_alg_errors.append(np.linalg.norm(X_hat[i, :] - test_set[i, :], ord=2))
        alg_error[alg] = np.mean(np.array(cur_alg_errors))
        alg_std[alg] = np.std(np.array(cur_alg_errors))

    print("Displaying the results for all algorithms...")
    for alg in range(num_algs):
        print("Alg: {0} Error: {1} Standard Deviation: {2}").format(algs[alg], alg_error[alg], alg_std[alg])

    #TODO: add the dot and 3d plots
    # print("Plotting the results...")
    # for i in range(num_costs):
    #     plt.title(graph_titles[i])
    #     plt.ylabel('Cost')
    #     plt.xlabel("Num Factors")
    #     plt.axis([1, num_factors, 0, 300])
    #     for j in range(num_algs):
    #          plt.plot([factor for factor in range(num_factors)], alg_costs[i][j], graph_colours[j], label="ALGO = {0}".format(algs[j]))
    #     plt.legend(loc='center', bbox_to_anchor=(0.60,0.90))
    #     plt.show()
    # print("Finished!")
