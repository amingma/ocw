
import numpy as np
import csv
import itertools, functools, operator

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    """ Return a d x 1 np array.
        value_list is a python list of values of length d.

    >>> cv([1,2,3])
    array([[1],
           [2],
           [3]])
    """
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    """ Return a 1 x d np array.
        value_list is a python list of values of length d.

    >>> rv([1,2,3])
    array([[1, 2, 3]])
    """
    return np.array([value_list])

# In all the following definitions:
# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar
def lin_reg(x, th, th0):
    """ Returns the predicted y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 0.]])
    >>> lin_reg(X, th, th0).tolist()
    [[1.05, 2.05, 3.05, 4.05]]
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> lin_reg(X, th, th0).tolist()
    [[3.05, 4.05, 5.05, 6.05]]
    """
    return np.dot(th.T, x) + th0

def square_loss(x, y, th, th0):
    """ Returns the squared loss between y_pred and y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> square_loss(X, Y, th, th0).tolist()
    [[4.2025, 3.4224999999999985, 5.0625, 3.8025000000000007]]
    """
    return (y - lin_reg(x, th, th0))**2

def mean_square_loss(x, y, th, th0):
    """ Return the mean squared loss between y_pred and y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> mean_square_loss(X, Y, th, th0).tolist()
    [[4.1225]]
    """
    # the axis=1 and keepdims=True are important when x is a full matrix
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True)

def ridge_obj(x, y, th, th0, lam):
    """ Return the ridge objective value

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> ridge_obj(X, Y, th, th0, 0.0).tolist()
    [[4.1225]]
    >>> ridge_obj(X, Y, th, th0, 0.5).tolist()
    [[4.623749999999999]]
    >>> ridge_obj(X, Y, th, th0, 100.).tolist()
    [[104.37250000000002]]
    """
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True) + lam * np.linalg.norm(th)**2

# Write a function that returns the gradient of lin_reg(x, th, th0)
# with respect to th
def d_lin_reg_th(x, th, th0):
    return x
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th.
def d_square_loss_th(x, y, th, th0):
    return -2*d_lin_reg_th(x, th, th0)*(y - lin_reg(x, th, th0))

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses d_square_loss_th.
def d_mean_square_loss_th(x, y, th, th0):
    return np.mean(d_square_loss_th(x, y, th, th0), axis=1, keepdims=True)

# Write a function that returns the gradient of lin_reg(x, th, th0)
# with respect to th0. Hint: Think carefully about what the dimensions of the returned value should be!
def d_lin_reg_th0(x, th, th0):
    return np.ones((1, x.shape[1]))
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th0.
def d_square_loss_th0(x, y, th, th0):
    return -2*(y - lin_reg(x, th, th0))

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses d_square_loss_th0.
def d_mean_square_loss_th0(x, y, th, th0):
    return np.mean(d_square_loss_th0(x, y, th, th0), axis=1, keepdims=True)

def d_ridge_obj_th(x, y, th, th0, lam):
    return d_mean_square_loss_th(x, y, th, th0) + 2*lam*th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return d_mean_square_loss_th0(x, y, th, th0)

#Concatenates the gradients with respect to theta and theta_0
def ridge_obj_grad(x, y, th, th0, lam):
    grad_th = d_ridge_obj_th(x, y, th, th0, lam)
    grad_th0 = d_ridge_obj_th0(x, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])    

# X: d by n data array
# y: 1 by n labels array
# J: objective function, takes in data point(cv), label, weight, and returns scalar
# dJ: cost function gradient, takes in same parameters as J and returns a cv
# w0: initial weight

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    n = X.shape[1]
    w = w0.copy()
    fs = [] # list of all Js found through all iterations
    ws = [] # list of all ws found through all iterations
    r = np.random.randint(n) # random int from 0 to n
    data = X[:, r:r+1]
    fs.append(J(data, y[:, r:r+1], w))
    ws.append(w)
    for i in range(max_iter-1):
        r = np.random.randint(n) # random int from 0 to n
        data = X[:, r:r+1]
        fs.append(J(data, y[:, r:r+1], w))
        w = w-step_size_fn(i)*dJ(data, y[:, r:r+1], w)
        ws.append(w)
    return w, fs, ws
        
    


############################################################
#From HW04; Used in the test case for sgd, below
def num_grad(f):
    def df(x):
        g = np.zeros(x.shape)
        delta = 0.001
        for i in range(x.shape[0]):
            xi = x[i,0]
            x[i,0] = xi - delta
            xm = f(x)
            x[i,0] = xi + delta
            xp = f(x)
            x[i,0] = xi
            g[i,0] = (xp - xm)/(2*delta)
        return g
    return df

#Test case for sgd
def sgdTest():
    def downwards_line():
        X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
        y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
        return X, y
    
    X, y = downwards_line()
    
    def J(Xi, yi, w):
        # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
        return float(ridge_obj(Xi[:-1,:], yi, w[:-1,:], w[-1:,:], 0))
    
    def dJ(Xi, yi, w):
        def f(w): return J(Xi, yi, w)
        return num_grad(f)(w)

    #Insert code to call sgd on the above
    pass

############################################################

def ridge_min(X, y, lam):
    """ Returns th, th0 that minimize the ridge regression objective
    
    Assumes that X is NOT 1-extended. Interfaces to our sgd by 1-extending
    and building corresponding initial weights.
    """
    def svm_min_step_size_fn(i):
        return 0.01/(i+1)**0.5

    d, n = X.shape
    X_extend = np.vstack([X, np.ones((1, n))])
    w_init = np.zeros((d+1, 1))

    def J(Xj, yj, th):
        return float(ridge_obj(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam))

    def dJ(Xj, yj, th):
        return ridge_obj_grad(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam)
    
    np.random.seed(0)
    w, fs, ws = sgd(X_extend, y, J, dJ, w_init, svm_min_step_size_fn, 1000)
    return w[:-1,:], w[-1:,:]

#######################################################################

def mul(seq):
    '''
    Given a list or numpy array of float or int elements, return the product 
    of all elements in the list/array.  
    '''
    return functools.reduce(operator.mul, seq, 1)

def make_polynomial_feature_fun(order):
    '''
    Transform raw features into polynomial features or order 'order'.
    If raw_features is a d by n numpy array, return a k by n numpy array 
    where k = sum_{i = 0}^order multichoose(d, i) (the number of all possible terms in the polynomial feature or order 'order')
    '''
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(1, order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f

######################################################################

#First finds a predictor on X_train and X_test using the specified value of lam
#Then runs on X_test, Y_test to find the RMSE
def eval_predictor(X_train, Y_train, X_test, Y_test, lam):
    th, th0 = ridge_min(X_train, Y_train, lam)
    return np.sqrt(mean_square_loss(X_test, Y_test, th, th0))

#Returns the mean RMSE from cross validation given a dataset (X, y), a value of lam,
#and number of folds, k
def xval_learning_alg(X, y, lam, k):
    '''
    Given a learning algorithm and data set, evaluate the learned classifier's score with k-fold
    cross validation. 
    
    learner is a learning algorithm, such as perceptron.
    data, labels = dataset and its labels.

    k: the "k" of k-fold cross validation
    '''
    _, n = X.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    X, y = X[:,idx], y[:,idx]

    split_X = np.array_split(X, k, axis=1)
    split_y = np.array_split(y, k, axis=1)

    score_sum = 0
    for i in range(k):
        X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=1)
        y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=1)
        X_test = np.array(split_X[i])
        y_test = np.array(split_y[i])
        score_sum += eval_predictor(X_train, y_train, X_test, y_test, lam)
    return score_sum/k

######################################################################
# For auto dataset

def load_auto_data(path_data):
    """
    Returns a list of dict with keys:
    """
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

# Feature transformations
def std_vals(data, f):
    '''
    Helper function to be used inside auto_data_and_labels. Returns average and standard deviation of 
    data's f-th feature. 
    >>> data = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    >>> f=0
    >>> std_vals(data, f)
    (3.5, 2.5)
    >>> f=3
    >>> std_vals(data, f)
    (6.5, 2.5)
    '''
    vals = [entry[f] for entry in data]
    avg = sum(vals)/len(vals)
    dev = [(entry[f] - avg)**2 for entry in data]
    sd = (sum(dev)/len(vals))**0.5
    return (avg, sd)

def standard(v, std):
    '''
    Helper function to be used in auto_data_and_labels. Center v by the 0-th element of std and scale by the 1-st element of std. 
    >>> data = np.array([1,2,3,4,5])
    >>> standard(data, (3,1))
    [array([-2., -1.,  0.,  1.,  2.])]
    >>> data = np.array([1,2,5,7,8])
    >>> standard(data, (3,1))
    [array([-2., -1.,  2.,  4.,  5.])]
    '''
    return [(v-std[0])/std[1]]

def raw(x):
    '''
    Make x into a nested list. Helper function to be used in auto_data_and_labels.
    >>> data = [1,2,3,4]
    >>> raw(data)
    [[1, 2, 3, 4]]
    '''
    return [x]

def one_hot(v, entries):
    '''
    Outputs a one hot vector. Helper function to be used in auto_data_and_labels.
    v is the index of the "1" in the one-hot vector.
    entries is range(k) where k is the length of the desired onehot vector. 

    >>> one_hot(2, range(4))
    [0, 0, 1, 0]
    >>> one_hot(1, range(5))
    [0, 1, 0, 0, 0]
    '''
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

# The class (mpg) added to the front of features
def auto_data_and_values(auto_data, features):
    features = [('mpg', raw)] + features
    std = {f:std_vals(auto_data, f) for (f, phi) in features if phi==standard}
    entries = {f:list(set([entry[f] for entry in auto_data])) \
               for (f, phi) in features if phi==one_hot}
    vals = []
    for entry in auto_data:
        phis = []
        for (f, phi) in features:
            if phi == standard:
                phis.extend(phi(entry[f], std[f]))
            elif phi == one_hot:
                phis.extend(phi(entry[f], entries[f]))
            else:
                phis.extend(phi(entry[f]))
        vals.append(np.array([phis]))
    data_labels = np.vstack(vals)
    return data_labels[:, 1:].T, data_labels[:, 0:1].T

######################################################################

#standardizes a row vector of y values
#returns both the standardized vector and the mean, variance
def std_y(row):
    '''
    >>> std_y(np.array([[1,2,3,4]]))
    (array([[-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079]]), array([2.5]), array([1.11803399]))
    '''
    mu = np.mean(row, axis=1)
    sigma = np.sqrt(np.mean((row - mu)**2, axis=1))
    return np.array([(val - mu)/(1.0*sigma) for val in row]), mu, sigma
