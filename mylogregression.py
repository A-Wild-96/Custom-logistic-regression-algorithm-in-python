import pandas as pd
import numpy as np

def sigmoid(z):
    '''
    Applies the sigmoid function to every element in input matrix/vector. 
    Parameters:
    ----------
    Input: z - numpy.ndarray of shape (N,)
    Output: g(z) - np.ndarray of shape (N,) - sigmoid function applied to each element of z
    '''
    g = 1 / (1 + np.exp(-z))
    return g

def calc_cost(X, y, W, b, lambda_):
    '''
    Calculates the cost function for a model applied to input features. The lower the cost, the better the model fits.
    The cost function can punish large parameter values for the model if reparameterisation (lambda_ != 0) is included.
    To get the true cost function of the data, do not include reparameterisation (lambda_ == 0).
    Parameters:
    ----------
    Input: X - pandas.core.frame.DataFrame shape (M,N) - features of dataset with M indices and N features
           y - pandas.core.frame.DataFrame shape (M,1) - target class of dataset with M indices
           W - numpy.ndarray of shape (N,) - feature parameters for model
           b - float - offset parameter for model
           lambda_ - float - reparameterisation parameter
    Output: cost - float - cost function of model with parameter W and b fitted to true data y with features X
    '''
    M = len(X.index)
    f_wb_vec = sigmoid(np.matmul(X.to_numpy(),W) + b)
    
    #Overflow issues with upcoming log's if entries are too close to zero or one
    f_wb_vec[f_wb_vec<1e-15]=1e-15
    f_wb_vec[1-f_wb_vec<1e-15]=1-1e-15
    
    loss_vec = -y.to_numpy().reshape(M,1)*np.log(f_wb_vec) - (1-y.to_numpy().reshape(M,1))*np.log(1 - f_wb_vec)

    
    cost = (sum(loss_vec)/M + (lambda_/(2*M))*np.sum(W*W))[0]
    return cost

def compute_gradient(X, y, W, b, lambda_):
    '''
    Computes the partial derivatives of the cost function with respect to model parameters.
    Parameters:
    ----------
    Input: X - pandas.core.frame.DataFrame shape (M,N) - features of dataset with M indices and N features
           y - pandas.core.frame.DataFrame shape (M,1) - target class of dataset with M indices
           W - numpy.ndarray of shape (N,1) - feature parameters for model
           b - float - offset parameter for model
           lambda_ - float - reparameterisation parameter
    Output:dj_db - float - partial derivative of cost function with respect to model parameter b
           dj_dW - numpy.ndarray of shape (N,1) - partial derivative of cost function wwith respect to model parameters in vector W
    '''
    (M,N) = X.shape
    dj_dW = np.array(np.zeros(N)).reshape(N,1)
    dj_db = 0.0
    
    f_wb_vec = sigmoid(np.matmul(X.to_numpy(),W) + b)
    dj_dW_vec = X.to_numpy()*(f_wb_vec - y.to_numpy().reshape(M,1))
    dj_db_vec = (f_wb_vec - y.to_numpy().reshape(M,1))
    dj_dW = np.sum(dj_dW_vec,axis=0).reshape(N,1)/M + lambda_*W/M
    dj_db = np.sum(dj_db_vec,axis=0)/M
    
    return dj_db, dj_dW

def gradient_descent(X, y, W_init, b_init, lambda_, alpha=1, max_iter=100, print_step=20, save_step=10):
    '''
    Perform gradient descent algorithm to minimise cost function by changing the model parameters.
    Parameters:
    ----------
    Input: X - pandas.core.frame.DataFrame shape (M,N) - features of dataset with M indices and N features
           y - pandas.core.frame.DataFrame shape (M,1) - target class of dataset with M indices
           W_init - numpy.ndarray shape (N,1) - initial feature parameters for model
           b_init - float - initial offset parameter for model
           alpha - float - learning rate of gradient descent
           max_iter - int - maximum number of iterations for gradient descent
           print_step - int - on iterations that are a multiple of print_step, print percentage of completion. If False do not print progress
           save_step - int - on iterations that are a multiple of save_step, store cost, models parameters and iterations in to vectors. If False only save last iteration
           lambda_ - float - reparameterisation parameter
    Output:iter_history - numpy.array - saved iterations for each multiple of save_step in max_iter 
           cost_reg_history - numpy.ndarray - cost function at saved iterations
           W - numpy.ndarray shape (N,1) - final model parameters
           b - float - final offset model parameter
           final_cost - float - final cost function of dataset (true cost does not use reparameterisation term) 
    '''
    
    #Initialisations
    cost_reg_history, iter_history, final_cost = [], [], 0.
    W, b = W_init, b_init
    for iter in range(1,max_iter+1):
        dj_db, dj_dW = compute_gradient(X, y, W, b, lambda_)   
        
        W = W - alpha*dj_dW
        b = b - alpha*dj_db
        
        
        cost_reg = calc_cost(X, y, W, b, lambda_)
        
        if(save_step!=0):
            if(iter % save_step or save_step==1):
                cost_reg_history.append(cost_reg)
                iter_history.append(iter) 
        elif(iter==max_iter):
            cost_reg_history.append(cost_reg)
            iter_history.append(iter) 
                
        if(print_step):
            if(iter % print_step == 0):print("{}%\n".format(100*iter/max_iter))
        
    #Calculate final cost of the model - note that this is the true cost
    #and so does not include reparameterisation (set lambda_=0.)    
    final_cost = calc_cost(X, y, W, b, lambda_=0.)
        
    return cost_reg_history, W, b, iter_history, final_cost

def min_max_scale(X):
    '''
    Applies min max scaling to input features.
    Parameters:
    ----------
    Input: X - pandas.core.frame.DataFrame - features of shape (M,N) with M indices and N features
    Output: X_minmax - pandas.core.frame.DataFrame - minmax scaled features of shape (M,N) with M indices and N features
    '''
    N = len(X.columns)
    X_minmax = X.copy(deep=True)
    for n in range(N):
        X_minmax.iloc[:,n] =  (X.iloc[:,n] - min(X.iloc[:,n]))/(max(X.iloc[:,n]) - min(X.iloc[:,n]))
    return X_minmax

def encode_polynomial(X, degree=1):
    '''
    Calculate polynomial features from input features.
    Parameters:
    ----------
    Input: X - pandas.core.frame.DataFrame - features of shape (M,N) with M indices and N features
           degree - float - maximum exponent applied to each feature
    Output: X_poly - pandas.core.frame.DataFrame - polynomial features of shape (M,N*degree) with M indices and N*degree features
    '''
    N = len(X.columns)
    N_poly = N*degree
    
    X_poly = X.copy(deep=True)

    for n in range(0,N):
        for d in range(1,degree+1):
            feat_label = "{}^{},".format(X.columns[n],d)
            #X_poly[feat_label] = X.iloc[:,n]**d
            
            new_feat = X.iloc[:,n].copy(deep=True)
            new_feat = new_feat**degree
            new_feat.rename(index=feat_label, inplace=True)
            X_poly = pd.concat([X_poly, new_feat], axis=1)
    X_poly.drop(labels=X.columns,axis=1,inplace=True)    
    
    return X_poly

def logistic_regression(X_train, X_test, y_train, y_test, lambda_, alpha=1, f_cutoff=0.5, degree=1, max_iter=100, print_step=20):
    '''
    Takes in input training data, uses gradient descent to find an optimal model, applies model to test features.
    After applying model to test features, estimated target class data is obtained which is compared to test target class data.
    Parameters:
    ----------
    Input: X_train - pandas.core.frame.DataFrame - training set features of shape (M_train,N) with M_train indices and N features
           y_train - pandas.core.frame.DataFrame - training set of shape (M_train,1) with target feature for each index
           X_test - pandas.core.frame.DataFrame - training set features of shape (M_test,N) with M_test indices and N features
           y_test - pandas.core.frame.DataFrame - training set of shape (M_test,1) with target feature for each index
           lambda_ - float - reparameterisation parameter
           alpha - float - learning rate of gradient descent
           f_cutoff - float - cutoff for sigmoid function to determine if y_hat is equal to 0 or 1.
           degree - int - powers of input features
           max_iter - int - maximum number of iterations for gradient descent
           print_step - int - on iterations that are a multiple of print_step, print percentage of completion. If False do not print progress
    Output:y_hat - numpy.array - predicted target features of shape (M_train,1) 
           cost_reg_history - numpy.ndarray - cost function at saved iterations
           W_model - numpy.ndarray shape (N,1) - final model parameters
           b_history - float - final offset model parameter
           final_cost - float - final cost function of dataset (true cost does not use reparameterisation term) 
    '''
    X_train = encode_polynomial(X_train,degree)
    X_train = min_max_scale(X_train)
    
    X_test = encode_polynomial(X_test,degree)
    X_test = min_max_scale(X_test)
    
    N_train = len(X_train.columns)
    W = np.ones(N_train).reshape(N_train,1)
    b = 0.
    cost_reg_history, W_model, b_model, print_steps, train_cost = gradient_descent(X_train, y_train, W, b, lambda_, alpha, max_iter, print_step, save_step=0)

    M_test = len(X_test.index)
    
    f_wb_vec = sigmoid(np.matmul(X_test.to_numpy(),W_model) + b_model)
    f_wb_vec[f_wb_vec >= f_cutoff] = 1
    f_wb_vec[f_wb_vec < f_cutoff] = 0
    
    y_hat = y_test.copy(deep=True)
    y_hat.iloc[:] = np.squeeze(f_wb_vec.reshape(1,M_test))
    
    test_cost = calc_cost(X_test, y_test, W_model, b_model, lambda_=0.)
                       
    return y_hat, W_model, b_model, train_cost, test_cost