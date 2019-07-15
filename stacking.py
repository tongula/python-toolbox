import numpy as np
import pandas as pd
from utility import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


def stacking_regression(models, df_encoded, y_train, n_linear, ver_no, meta_model= None, n_folds=5, average_fold=True):
    '''
    Function 'stacking_regression' takes list of 1-st level models, train and test set,
    and train a 2-nd level meta model if provided or save the model predictions.

    Parameters
    ----------
    models : list
        List of 1-st level models. You can use any models that follow sklearn
        convention i.e. accept numpy arrays and have methods 'fit' and 'predict'.
        Must be of form [tree model 1, tree model 2, ..., tree model n, linear model 1, linear model 2, ..., linear model m]

    df_encoded : dataframe
        Include both train set (without label) and test set. Could be separated with 'Id'.

    y_train : dataframe
        Target values

    n_linear : int
        Number of linear models in the list of models.

    ver_no : int
        Version number for bookkeeping.

    meta_model: model, default None
        If None, no 2nd layer to be trained. Only save the predictions from cross validations in './stacking_submissions/raw/'
        If a model is provided, a 2nd level model is trained and predictions are saved for submission.
        You can use any model that follow sklearn convention

    n_folds : int, default 5
        Number of folds in cross-validation

    average_fold: boolean, default True
        Whether to take the average of the predictions on test set from each fold.
        Refit the model using the whole training set and predict test set if False
    '''

    # Metric used to display.
    metric = mean_squared_error

    X_train = df_encoded[df_encoded['Id']<=1460]
    X_test = df_encoded[df_encoded['Id']>1460]
    X_train.drop('Id',axis=1,inplace=True)
    X_test.drop('Id',axis=1,inplace=True)

    # need to normalize for linear models
    X_train_lin , X_test_lin = normalizeDf(X_train, X_test)
    # transform dataframe to nd.matrix works better with KFold
    X_train = X_train.values
    X_test = X_test.values
    X_train_lin = X_train_lin.values
    X_test_lin = X_test_lin.values

    # reset index works better with KFold
    y_train = y_train.reset_index(drop=True)

    # Split indices to get folds
    kf = KFold(n_splits = n_folds, shuffle = True, random_state = 42)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))

    # Loop across models
    for model_counter, model in enumerate(models):

        print('model %d: [%s]' % (model_counter+1, model.__class__.__name__))
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Loop across folds
        if model_counter == len(models)-n_linear:
            X_train,X_test = X_train_lin,X_test_lin
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            # Clone the model because fit will mutate the model.
            instance = clone(model)
            # Fit 1-st level model
            instance.fit(X_tr, y_tr)
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = instance.predict(X_te)
            # Predict full test set
            S_test_temp[:, fold_counter] = instance.predict(X_test)
            # Delete temperatory model
            del instance

        # Compute mean or mode of predictions for test set
        if average_fold:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            model.fit(X_train, y_train)
            S_test[:, model_counter] = model.predict(X_test)

        print('    ----')
        print('    MEAN:   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    if not meta_model:
        print('Logging model results in ./stacking_submissions/raw/')
        # Save S_train, S_test for later use
        S_tr = pd.DataFrame(S_train)
        S_te = pd.DataFrame(S_test)
        S_tr.to_csv('./stacking_submissions/raw/stack_raw_train'+str(ver_no)+'.csv',index=False)
        S_te.to_csv('./stacking_submissions/raw/stack_raw_test'+str(ver_no)+'.csv',index=False)

    else:
        print('Training 2nd layer meta model.')
        # Fit our second layer meta model
        meta_model.fit(S_train, y_train)
        final_score = metric(y_train,meta_model.predict(S_train))
        # Make our final prediction
        stacking_prediction = meta_model.predict(S_test)
        stacking_prediction = np.expm1(stacking_prediction)
        submission = pd.DataFrame({'Id':list(range(1461,2920)),'SalePrice':stacking_prediction})
        submission.to_csv('./stacking_submissions/stack'+str(ver_no)+'_'+str(final_score)+'.csv',index=False)
