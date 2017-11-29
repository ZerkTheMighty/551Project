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
    #TODO: aa and kmeans are throwing errors when attempting to run it; look into this later
    algs = ['pca', 'cmeans']
    num_algs = len(algs)
    #Corresponds to temperature and age, respectively
    float_to_int_idxs = [7, 13]
    #Corresponds to sex and nerve type (leg or arm), respectively
    float_to_binary_idxs = [14, 35]

    #TODO: Delete data properly in the test set (make sure it is representative of the proportions in the original data set)
    #NOTE: Based on the documentation in the class it looks like W is the dictionary, and H is the samples
    #so it looks like we have to transpose our data matrix: don't know how this will impact the cost function formulas
    all_data, trainset, testset = dtl.load_nerve_data(trainsize, testsize)
    for alg in range(num_algs):
        cur_alg = algs[alg]
        for k in range(num_factors):
            print(('Running on train={0} and test={1} samples for algorithm: {2}, with {3} factors').format(trainset.shape[0], testset.shape[0], cur_alg, k))

            if cur_alg == 'pca':
                cur_mdl = pca.PCA(np.transpose(all_data), num_bases=36)
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


            #print(np.transpose(all_data)[7, :])
            print(np.transpose(all_data)[13, :])
            #print(np.transpose(all_data)[14, :])
            #print(np.transpose(all_data)[35, :])

            X_hat = np.dot(cur_mdl.W, cur_mdl.H)
            #Round certain features as appropriate after factorizing
            #print(X_hat[7, :])
            print(X_hat[13, :])
            #print(X_hat[14, :])
            #print(X_hat[35, :])
            sanitize_features(X_hat, float_to_int_idxs, float_to_binary_idxs)
            #print(X_hat[7, :])
            print(X_hat[13, :])
            #print(X_hat[14, :])
            #print(X_hat[35, :])
            exit()

            #TODO: Compute the relevant costs and store them for plotting
            # if cur_alg == 'svd':
            #     break

    #TODO: Modify this to appropriately plot the results
    # print "\nPlotting the results..."
    # plt.ylabel('K Factors')
    # plt.xlabel("Cost")
    # plt.axis([1, num_episodes, 0, 1])
    # for i in range(len(avg_results)):
    #     plt.plot([episode for episode in range(num_episodes)], avg_results[i], GRAPH_COLOURS[i], label="Alpha = " + str(ALPHAS[i]) + " AGENT = " + str(AGENTS[i]))
    # plt.legend(loc='center', bbox_to_anchor=(0.60,0.90))
    # plt.show()
    # print "\nFinished!"
