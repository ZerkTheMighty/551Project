from __future__ import division  # floating point division
import numpy as np
import utilities as utils
from math import pi, sqrt, exp, log

class Classifier(object):
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.numclasses = 2

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = parameters
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = parameters
        self.reset(parameters)
        self.numclasses = 2

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
        #We use the range function throughout, so we label the first feature as 0 in many of our loops
        if self.params['usecolumnones']:
            self.numfeatures = Xtrain.shape[1] - 1
        else:
            self.numfeatures = Xtrain.shape[1] - 2
        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)

        ### YOUR CODE HERE
        #Gather the class counts and feature instances per class
        Xtrain = Xtrain[:, 0:self.numfeatures]
        class_counts = [0 for data_class in range(self.numclasses)]
        feature_instances = [[[] for feature in range(self.numfeatures)] for data_class in range(self.numclasses)]
        for i in range(Xtrain.shape[0]):
            cur_class = int(ytrain[i])
            class_counts[cur_class] += 1
            for feature in range(self.numfeatures):
                feature_instances[cur_class][feature].append(Xtrain[i][feature])

        #We compute the mean and std here from the whole set instead of as we go so we can use numpy, to avoid arithmetical issues
        for data_class in range(self.numclasses):
            for feature in range(self.numfeatures):
                self.means[data_class][feature] = np.mean(feature_instances[data_class][feature])
                self.stds[data_class][feature] = np.std(feature_instances[data_class][feature])

        #Compute the class prior probabilites
        self.class_probs = [class_count / Xtrain.shape[0] for class_count in class_counts]
        ### END YOUR CODE

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        ytest = []
        for target in Xtest:
            #The elements of the product of probabilites for each class
            prob_prod_elements = [[self.gaussian(target[feature], self.means[data_class][feature], self.stds[data_class][feature]) for feature in range(self.numfeatures)] + [self.class_probs[data_class]] for data_class in range(self.numclasses)]
            final_probs = map(np.prod, prob_prod_elements)
            ytest.append(final_probs.index(max(final_probs)))
        ytest = np.array(ytest, dtype=int)
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

    def gaussian(self, x, mean, sd):
        var = float(sd) ** 2
        denom = sqrt((2 * pi * var))
        num = exp(-(float(x)-float(mean)) ** 2 / (2 * var))
        return num / denom

class LogitReg(Classifier):

    def __init__(self, parameters={'regularizer': 'l2'}):
        # Default: no regularization
        self.params = parameters
        self.reset(parameters)
        self.numclasses = 2

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        ### YOUR CODE HERE
        epsilon = 1.0e-6 # To avoid trying to take the log of 0 or something too numerically close to it
        hypothesis = utils.sigmoid(np.dot(X, theta))
        numsamples = X.shape[0]
        if 'regwgt' in self.params:
            cost = (sum((-y * map(log, hypothesis + epsilon)) - ((1 - y) * map(log, 1 - hypothesis + epsilon)))) / numsamples + ((self.params['regwgt'] / (2 * numsamples)) * sum(theta ** 2))
        else:
            cost = (sum((-y * map(log, hypothesis + epsilon)) - ((1 - y) * map(log, 1 - hypothesis + epsilon)))) / numsamples
        ### END YOUR CODE
        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE
        hypothesis = utils.sigmoid(np.dot(X, theta))
        numsamples = X.shape[0]
        if 'regwgt' in self.params:
            for i in range(len(grad)):
                grad[i] = sum((hypothesis - y) * X[:, i]) / numsamples + ((self.params['regwgt'] / numsamples) * theta[i])
        else:
            for i in range(len(grad)):
                grad[i] = sum((hypothesis - y) * X[:, i]) / numsamples
        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1])
        ### YOUR CODE HERE
        #We learn via batch gradient descent
        cur_cost = float("inf")
        tolerance = 10 ** -10
        new_cost = self.logit_cost(self.weights, Xtrain, ytrain)
        while abs(new_cost - cur_cost) > tolerance:
            cur_cost = new_cost
            gradient = self.logit_cost_grad(self.weights, Xtrain, ytrain)
            step_size = self.line_search(self.weights, cur_cost, gradient, Xtrain, ytrain, Xtrain.shape[0])
            self.weights -= (step_size * gradient)
            new_cost = self.logit_cost(self.weights, Xtrain, ytrain)
            print("Logistic Regression Cost: " + str(new_cost))
        ### END YOUR CODE

    def line_search(self, weights, cost, gradient, X, y, numsamples):

        step_size_max = 10.00
        step_size_reducer = 0.50
        tolerance = 10e-4
        max_iterations = 1000

        cur_step_size = step_size_max
        cur_weights = weights
        cur_obj = cost
        for i in range(max_iterations):
            cur_weights = weights - (cur_step_size * gradient)
            new_cost = self.logit_cost(cur_weights, X, y)
            if new_cost < (cur_obj - tolerance):
                return cur_step_size
            else:
                cur_step_size = step_size_reducer * cur_step_size
                cur_obv = new_cost
        return 0

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        regression_values = utils.sigmoid(np.dot(Xtest, self.weights));
        for i in range(len(regression_values)):
            if regression_values[i] >= 0.5:
                ytest[i] = 1
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 10}
        self.reset(parameters)
        self.numfeatures = 9
        self.numclasses = 2

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_hidden = self.transfer(np.dot(self.w_input, inputs))

        # output activations
        a_output = self.transfer(np.dot(self.w_output, a_hidden))

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE
        #Ensure that the input is a proper 2 dimensional matrix
        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        #Perform a forward pass, getting the outputs for each layer
        hidden_layer_results, output_layer_results = self.feedforward(x)

        #Compute the error for the output layer
        output_error = output_layer_results - y

        #Compute the error for the hidden input_layer_results
        hidden_layer_error = np.dot(np.transpose(self.w_output), output_error)

        #Compute the gradients
        nabla_input = np.dot(hidden_layer_error, np.transpose(x))
        nabla_output = np.dot(output_error, np.transpose(hidden_layer_results))
        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        #Initialize the weights randomly to make sure that hidden units do not all learn the same weights
        self.w_input = np.random.randn(self.params['nh'], Xtrain.shape[1]) #self.numfeatures in place of Xtrain
        self.w_output = np.random.randn(1, self.params['nh'])
        numsamples = Xtrain.shape[0]
        ytrain = ytrain[:, np.newaxis]
        epochs = self.params['epochs']
        costs = []
        num_iterations = 0
        for epoch in range(epochs):
            print("Stochastic epoch: " + str(epoch))
            #Shuffle the data, making sure to maintain the proper correspondence between the features and targets
            data_set = np.append(Xtrain, ytrain, axis=1)
            np.random.shuffle(data_set)
            Xtrain = data_set[:, 0:data_set.shape[1] - 1]
            ytrain = data_set[:, -1, np.newaxis]
            for sample in range(numsamples):
                cur_sample = np.transpose(Xtrain[sample, np.newaxis])
                cur_target = ytrain[sample, np.newaxis]
                costs.append(self.cost(cur_sample, cur_target))
                num_iterations += 1
                #Check that SGD is converging
                if num_iterations == 5000:
                    print('Average neural network cost over the previous epoch')
                    print(np.mean(np.array(costs)))
                    num_iterations = 0
                    costs = []

                #Update the weights
                input_gradient, output_gradient = self.backprop(cur_sample, cur_target)
                step_size = self.params['stepsize'] / (epoch + 1)
                self.w_input -= (step_size * input_gradient)
                self.w_output -= (step_size * output_gradient)

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        numsamples = Xtest.shape[0]
        ytest = np.zeros(numsamples, dtype=int)
        for i in range(numsamples):
            if self.feedforward(np.transpose(Xtest[i, :]))[1] >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0
        return ytest

    def cost(self, x, y):
        "Return the cost on data set X given the current weights"
        ### YOUR CODE HERE
        epsilon = 1.0e-6 #Used to prevent errors due to numerical instability close to 0
        hypothesis = self.feedforward(x)[1]
        cost = (sum((-y * map(log, hypothesis + epsilon)) - ((1 - y) * map(log, 1 - hypothesis + epsilon))))
        return cost
        ### END YOUR CODE


class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = parameters
        self.reset(parameters)
        self.centers = None
        self.numclasses = 2

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None

        ### YOUR CODE HERE
        Ktrain = self.transform_data(Xtrain, self.params['kernel'], self.params['num_centers'])
        ### END YOUR CODE

        ### YOUR CODE HERE
        self.weights = np.zeros(Ktrain.shape[1])
        cur_cost = float("inf")
        tolerance = 10 ** -6
        new_cost = self.logit_cost(self.weights, Ktrain, ytrain)

        #Set the step size based on the kernel type and the number of features
        #We do it here rather than as a parameter because deviating too far from
        # these step sizes for the given number of centers tends to result in divergence
        if self.params['kernel'] == 'linear':
            if self.params['num_centers'] == 100:
                step_size = 0.07 #100 features
            else:
                step_size = 0.001 #1000 and 2500 features
        elif self.params['kernel'] == 'hamming':
            step_size = 0.001 #Both 100 and 1000 features (centers)

        while abs(new_cost - cur_cost) > tolerance:
            cur_cost = new_cost
            gradient = self.logit_cost_grad(self.weights, Ktrain, ytrain)
            self.weights -= (step_size * gradient)
            new_cost = self.logit_cost(self.weights, Ktrain, ytrain)
            print('Logistic Regression Cost: ' + str(new_cost))
            print('Cost change after wight update: ' + str(abs(new_cost - cur_cost)))
            print('Tolerance to stop the descent: ' + str(tolerance))
        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.


    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE

        Ktest = self.transform_data(Xtest, self.params['kernel'], self.params['num_centers'])
        regression_values = utils.sigmoid(np.dot(Ktest, self.weights))
        for i in range(len(regression_values)):
            if regression_values[i] >= 0.5:
                ytest[i] = 1
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

    def transform_data(self, X, kernel, num_centers):
        "Transforms the input data according to the given kernel"
        if kernel != 'None':
            numsamples = X.shape[0]
            if kernel == 'hamming':
                X = X[:, np.newaxis]
            if self.centers != None:
                kernel_centers = self.centers
            else:
                kernel_centers = X[0:num_centers, :]
                self.centers = kernel_centers
            Ktrain = np.zeros((numsamples, num_centers))
            for i in range(numsamples):
                #Row vector
                if kernel == 'linear':
                    cur_feature_vec = X[i, :][np.newaxis]
                elif kernel == 'hamming':
                    cur_feature_vec = X[i, :]
                else:
                    exit("Invalid kernel " + kernel + " specified. Must be one of 'linear', 'hamming', 'None'")
                #Column vector
                new_feature_vec = np.zeros((1, num_centers))
                for j in range(num_centers):
                    if kernel == 'linear':
                        #Kernel centre at J is 1 X 9
                        new_feature_vec[0][j] = np.dot(cur_feature_vec, np.transpose(kernel_centers[j]))
                    elif kernel == 'hamming':
                        new_feature_vec[0][j] = self.hamming_distance(cur_feature_vec, np.transpose(kernel_centers[j]))
                    else:
                        exit("Invalid kernel " + kernel + " specified. Must be one of 'linear', 'hamming', 'None'")
                Ktrain[i, :] = new_feature_vec
            return Ktrain
        return X

    def hamming_distance(self, s1, s2):
        """Return the Hamming distance between s1 and s2"""
        if len(s1) != len(s2):
            raise ValueError("Invalid input: s1 and s2 must be of equal length!")
        return sum(el1 != el2 for el1, el2 in zip(s1[0], s2[0]))


# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"
    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)
    print("BACKPROP TEST")
    res = clf.backprop(np.transpose(X[0, :]), y[0])
    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(np.transpose(X[0, :]), y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()


if __name__ == "__main__":
    main()
