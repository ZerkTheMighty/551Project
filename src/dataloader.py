from __future__ import division  # floating point division
import math
import numpy as np

####### Main load functions
def load_nerve_data(trainsize, testsize):
    """Electrodiagnostic nerve test data """
    filename = '../datasets/nerveData.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset, trainsize, testsize, outputfirst=True)
    return trainset, testset

####### Helper functions
def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    # Generate random indices without replacement, to make train and test sets disjoint
    randindices = np.random.choice(dataset.shape[0], trainsize+testsize, replace=False)

    Xtrain = dataset[randindices[0:trainsize], :]
    Xtest = dataset[randindices[trainsize:trainsize+testsize], :]

    #TODO: Consider if we want to do this for our data set. My first reaction is no, since I'm guessing the code below
    #assumes a supervised prediction problem

    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility

    # for ii in range(Xtrain.shape[1]):
    #     maxval = np.max(np.abs(Xtrain[:,ii]))
    #     if maxval > 0:
    #         Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
    #         Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)

    return (Xtrain, Xtest)
