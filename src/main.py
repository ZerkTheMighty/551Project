from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import dataloader as dtl
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from pymf import pca, cmeans, aa, kmeans, svd


def sanitize_features(X, float_to_int_idxs, float_to_binary_idxs):
    """
    Sanitize the matrix X by rounding floats to ints and thresholding floats
    to binary variables at the provided indices of X's feature vectors.
    Assumes X is of the format features X samples
    """
    num_samples = X.shape[1]
    for index in float_to_int_idxs:
        for sample in range(num_samples):
            X[index, sample] = int(round(X[index, sample]))

    for index in float_to_binary_idxs:
        for sample in range(num_samples):
            cur_val = X[index, sample]
            if cur_val >= 0.5:
                X[index, sample] = 1
            else:
                X[index, sample] = 0


if __name__ == '__main__':
    trainsize = 125
    testsize = 40
    num_factors = 36 #Including the binary arm/leg distinction where 1 = leg and 0 = arm
    graph_colours = ['r', 'g', 'b']
    graph_titles = ['Test set contains missing values', "Test set without missing values", "Full test set"]
    algs = ['pca', 'cmeans', 'aa']
    num_algs = len(algs)
    num_costs = 3
    alg_costs = [[[] for cost in range(num_algs)] for alg in range(num_costs)]

    #Corresponds to temperature and age, respectively
    float_to_int_idxs = [7, 13]
    #Corresponds to sex and nerve type (leg or arm), respectively
    float_to_binary_idxs = [14, 35]


    #TODO: Delete data properly in the test set (make sure it is representative of the proportions in the original data set)
    #NOTE: Based on the documentation in the class it looks like W is the dictionary, and H is the samples
    #so it looks like we have to transpose our data matrix: don't know how this will impact the cost function formulas as written in the draft
    all_data, trainset, testset = dtl.load_nerve_data(trainsize, testsize)
    for alg in range(num_algs):
        cur_alg = algs[alg]
        for k in range(1, num_factors + 1):
            print(('Running on train={0} and test={1} samples for algorithm: {2}, with {3} factors').format(trainset.shape[0], testset.shape[0], cur_alg, k))

            if cur_alg == 'pca':
                cur_mdl = pca.PCA(np.transpose(all_data), num_bases=k)
                cur_mdl.factorize()
            elif cur_alg == 'cmeans':
                cur_mdl = cmeans.Cmeans(np.transpose(all_data), num_bases=k)
                cur_mdl.factorize()
            elif cur_alg == 'aa':
                cur_mdl = aa.AA(np.transpose(all_data), num_bases=k)
                cur_mdl.factorize()
            elif cur_alg == 'kmeans':
                cur_mdl = kmeans.Kmeans(np.transpose(all_data), num_bases=k)
                cur_mdl.factorize()
            elif cur_alg == 'svd':
                cur_mdl = svd.SVD(np.transpose(all_data))
                cur_mdl.factorize()
            else:
                exit("ERROR: Invalid algorithm {0} selected!!!".format(cur_alg))

            X_hat = np.dot(cur_mdl.W, cur_mdl.H)
            #Round certain features as appropriate after factorizing
            sanitize_features(X_hat, float_to_int_idxs, float_to_binary_idxs)

            #TODO: Compute the relevant costs and store them for plotting:
            #Need to ensure that the score for the sanitized X is used: currently uses just the forbenius norm of the unsanitized matrix
            for i in range(num_costs):
                # print(cur_mdl.W.shape)
                # print(cur_mdl.H.shape)
                # print(np.dot(cur_mdl.W, cur_mdl.H))
                #TODO: This is returning a matrix for cmeans for some reason...look into this
                alg_costs[i][alg].append(cur_mdl.ferr)

    print("Plotting the results...")
    for i in range(num_costs):
        plt.title(graph_titles[i])
        plt.ylabel('Cost')
        plt.xlabel("Num Factors")
        plt.axis([1, num_factors, 0, 300])
        for j in range(num_algs):
             plt.plot([factor for factor in range(num_factors)], alg_costs[i][j], graph_colours[j], label="ALGO = {0}".format(algs[j]))
        plt.legend(loc='center', bbox_to_anchor=(0.60,0.90))
        plt.show()
    print("Finished!")
