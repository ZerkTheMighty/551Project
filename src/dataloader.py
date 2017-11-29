from __future__ import division  # floating point division
import math
import numpy as np

####### Main load functions
def load_blog(trainsize=5000, testsize=5000):
    """ A blogging dataset """
    if trainsize + testsize < 5000:
        filename = 'datasets/blogData_train_small.csv'
    else:
        filename = 'datasets/blogData_train.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize,featureoffset=50)
    return trainset,testset

def load_ctscan(trainsize=5000, testsize=5000):
    """ A CT scan dataset """
    if trainsize + testsize < 5000:
        filename = 'datasets/slice_localization_data.csv'
    else:
        filename = 'datasets/slice_localization_data.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize,featureoffset=1)
    return trainset,testset

def load_song(trainsize=5000, testsize=5000):
    """ The million song dataset
     Not a good dataset for feature selection or regression
     Standard linear regression performs only a little bit better than a random vector.
     Additional complex models, such as interesting kernels, are needed
     To improve performance
     """
    if trainsize + testsize < 5000:
        filename = 'datasets/YearPredictionMSD_small.csv'
    else:
        filename = 'datasets/YearPredictionMSD.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize,outputfirst=True)
    return trainset,testset

def load_susy(trainsize=500, testsize=1000):
    """ A physics classification dataset """
    filename = 'datasets/susysubset.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize)
    return trainset,testset

def load_susy_complete(trainsize=500, testsize=1000):
    """ A physics classification dataset """
    filename = 'datasets/susycomplete.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize,outputfirst=True)
    return trainset,testset

def load_census(trainsize=1000, testsize=1000):
    maxsamples = 32562;
    strtype = 'a50'
    censusdtype={'names': ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'), 'formats': ('f', strtype, 'f', strtype,'f',strtype,strtype,strtype,strtype,strtype,'f','f','f',strtype,strtype)};
    incomeindex = 14;
    convs = {14: lambda s: int(b'=' in s)};
    dataset = np.loadtxt('datasets/censusincome.txt', delimiter=',',dtype=censusdtype,converters = convs)
    numsamples = dataset.shape[0]
    subsetsamples =  trainsize+testsize

    # Doing this specifically for census, since we do not want to add a bias unit
    # and because we cannot normalize features
    randindices = np.random.choice(numsamples,subsetsamples, replace=False)
    vals = np.zeros(subsetsamples);
    for ii in range(subsetsamples):
        if b'1' == dataset[randindices[ii]][incomeindex]:
            vals[ii] = 1.0

    Xtrain = dataset[randindices[0:trainsize]]
    ytrain = vals[0:trainsize]
    Xtest = dataset[randindices[trainsize:trainsize+testsize]]
    ytest = vals[trainsize:trainsize+testsize]

    # Remove the targets from Xtrain and Xtest
    allfeatures = list(censusdtype['names'][0:incomeindex])
    Xtrain = Xtrain[allfeatures]
    Xtest = Xtest[allfeatures]

    return ((Xtrain,ytrain), (Xtest,ytest))

def load_madelon():
    datasettrain = np.genfromtxt('datasets/madelon/madelon_train.data', delimiter=' ')
    trainlab = np.genfromtxt('datasets/madelon/madelon_train.labels', delimiter=' ')
    trainlab[trainlab==-1] = 0
    trainsetx = np.hstack((datasettrain, np.ones((datasettrain.shape[0],1))))
    trainset = (trainsetx, trainlab)

    datasettest = np.genfromtxt('datasets/madelon/madelon_valid.data', delimiter=' ')
    testlab = np.genfromtxt('datasets/madelon/madelon_valid.labels', delimiter=' ')
    testlab[testlab==-1] = 0
    testsetx = np.hstack((datasettest, np.ones((datasettest.shape[0],1))))
    testset = (testsetx, testlab)

    return trainset,testset

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
    #np.random.seed(123)
    randindices = np.random.choice(dataset.shape[0],trainsize+testsize, replace=False)
    featureend = dataset.shape[1]-1
    outputlocation = featureend
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize],featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize],outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize+testsize],outputlocation]

    if testdataset is not None:
        Xtest = dataset[:,featureoffset:featureend]
        ytest = dataset[:,outputlocation]

    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        if maxval > 0:
            Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
            Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))

    return ((Xtrain,ytrain), (Xtest,ytest))


def create_susy_dataset(filenamein,filenameout,maxsamples=100000):
    dataset = np.genfromtxt(filenamein, delimiter=',')
    y = dataset[0:maxsamples,0]
    X = dataset[0:maxsamples,1:9]
    data = np.column_stack((X,y))

    np.savetxt(filenameout, data, delimiter=",")
