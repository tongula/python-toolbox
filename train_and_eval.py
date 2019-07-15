import xgboost as xgb
import math
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from utility import *


# Need to scale for linear models
def train_elastic(df_encoded, y_train, ver_no, ratio = [0.01,0.25,0.5,0.75,1],n_alphas = 1000):

    # ----Seperate df_encoded into train and test and drop Id column----
    x_train = df_encoded[df_encoded['Id']<=1460]
    x_test = df_encoded[df_encoded['Id']>1460]
    x_train.drop('Id',axis=1,inplace=True)
    x_test.drop('Id',axis=1,inplace=True)

    # ----Normalize----
    x_train, x_test = normalizeDf(x_train, x_test)

    # ----Cross Validate to get the best ration and best Alpha ----
    model = ElasticNetCV(l1_ratio=ratio, n_alphas=n_alphas, cv=10, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    # Record best alpha and l1_ratio:
    alpha = model.alpha_
    print('Chosen alpha: ', alpha)
    l1_ratio = model.l1_ratio_
    print('Chosen l1_ratio: ', l1_ratio)
    print('1 means lasso, 0 means ridge')
    # ----Use best hyper parameeters to train the final model----
    modelCalibrated = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
    modelCalibrated.fit(x_train, y_train)

    # ----Predict the output for submission and model scoring----
    y_pred_test = modelCalibrated.predict(x_test)
    y_pred_test = np.expm1(y_pred_test)
    y_pred_train = modelCalibrated.predict(x_train)

    # ----Calculate Log RMSE in-sample----

    rmse = math.sqrt(sum((y_train-y_pred_train)**2)/len(x_train))
    print('Train RMSE: ', rmse)

    # ----Return Feature Importance----
    features = list(zip(list(x_train.columns.values),list(modelCalibrated.coef_)))
    featureImportance = pd.DataFrame(features).sort_values(by =1,ascending=False)
    featureImportance = featureImportance.rename(columns = {0:'Feature',1:'Coef'})
    featureImportance = featureImportance[featureImportance['Coef']!=0]

    # ----Save Submisssion----
    submission = pd.DataFrame({'Id':list(range(1461,2920)),'SalePrice':y_pred_test})
    file_name = './submissions/ver_'+str(ver_no)+'_elasticnet_'+str(l1_ratio)+'_'+str(alpha)+'_'+str(rmse)+'.csv'
    submission.to_csv(file_name,index=False)

    # ----Create DataFrame of Predicted vs. Actual SalePrice----
    diff = pred_vs_actual(df_encoded, y_pred_train, y_train)

    return diff, featureImportance, l1_ratio, alpha


# Don't need to scale for tree models
def train_rf(df_encoded,y_train,ver_no, n_estimators,max_features,max_depth,oob_score=True):

    x_train = df_encoded[df_encoded['Id']<=1460]
    x_test = df_encoded[df_encoded['Id']>1460]
    x_train.drop('Id',axis=1,inplace=True)
    x_test.drop('Id',axis=1,inplace=True)

    rf = RandomForestRegressor(n_estimators = n_estimators,max_features=max_features,max_depth=max_depth,
                            oob_score=oob_score,random_state = 42,n_jobs=-1)

    rf.fit(x_train, y_train)

    y_pred_train = rf.predict(x_train)
    y_pred_test = rf.predict(x_test)
    y_pred_test = np.expm1(y_pred_test)

    rmse = math.sqrt(sum((y_train-y_pred_train)**2)/len(x_train))
    print('Rmse: ',rmse)


    if oob_score:
        oob = rf.oob_score_
        print('Oob score: ',oob)
        oob_pred = rf.oob_prediction_
        rmse_oob = math.sqrt(sum((y_train-oob_pred)**2)/len(x_train))
        print('Rmse using oob prediction: ', rmse_oob)

    #Save Submisssion-------------------------------------------------------------------------------------------------------------------------
    file_name = './submissions/ver_'+str(ver_no)+'_rf_'+str(n_estimators)+'_'+str(max_features)+'_'+str(max_depth)+'_'+str(rmse)+'.csv'
    submission = pd.DataFrame({'Id':list(range(1461,2920)),'SalePrice':y_pred_test})
    submission.to_csv(file_name,index=False)

    #Return Feature Importance-------------------------------------------------------------------------------------------------------------------------
    features = list(zip(list(x_train.columns.values),list(rf.feature_importances_)))
    featureImportance = pd.DataFrame(features).sort_values(by =1,ascending=False)
    featureImportance = featureImportance.rename(columns = {0:'Feature',1:'Frequency_in_Splits'})
    featureImportance = featureImportance[featureImportance['Frequency_ini_Splits']!=0]

    return featureImportance,rmse,rmse_oob



def train_xgb(df_encoded,y_train,ver_no,lr_list,depth_list,gamma_list,lambda_list,max_steps=1000):

    x_train = df_encoded[df_encoded['Id']<=1460]
    x_test = df_encoded[df_encoded['Id']>1460]
    x_train.drop(['Id'],axis=1,inplace=True)
    x_test.drop(['Id'],axis=1,inplace=True)

    dtrain = xgb.DMatrix(x_train, y_train)

    # first , tune learning_rate:
    # all other params are set to default
    best_lr = lr_list[0]

    if(len(lr_list)>1):
        print("Tuning learning rate:")
        best_score = 1e7
        for lr in lr_list:
            params = {'tree_method': 'gpu_hist'}
            params['learning_rate']= lr
            num_round = int(10/lr)
            k_fold = model_selection.KFold(n_splits=10, shuffle=True,random_state=42)
            cv_metrics = xgb.cv(params, dtrain, num_boost_round=num_round,folds=k_fold, callbacks=[xgb.callback.print_evaluation(), xgb.callback.early_stop(3)])
            score = min(cv_metrics.iloc[:,2])

            if score<best_score:
                best_lr = lr
                best_score = score
        print("Tuning learning rate done! Chose: ", best_lr)

    # next, tune other hyper parameters:
    best_dep=depth_list[0]
    best_gamma=gamma_list[0]
    best_lambda=lambda_list[0]
    if len(depth_list)>1 or len(gamma_list)>1 or len(lambda_list)>1:
        print("Tuning depth,gamma and lambda:")
        best_score = 1e7
        for dep in depth_list:
            for gam in gamma_list:
                for lam in lambda_list:
                    params = {'tree_method': 'gpu_hist','learning_rate':best_lr}
                    params['max_depth']=dep
                    params['gamma']=gam
                    params['lambda']=lam
                    num_round = 50
                    k_fold = model_selection.KFold(n_splits=10, shuffle=True,random_state=42)
                    cv_metrics = xgb.cv(params, dtrain, num_boost_round=num_round,folds=k_fold, callbacks=[xgb.callback.print_evaluation(), xgb.callback.early_stop(3)])
                    score = min(cv_metrics.iloc[:,2])

                    if score<best_score:
                        best_dep = dep
                        best_gamma=gam
                        best_lambda=lam
                        best_score = score

    # Train the model with best hyperparameters chosen from cv
    params = {'tree_method': 'gpu_hist'}
    params['max_depth']=best_dep
    params['learning_rate']=best_lr
    params['gamma']=best_gamma
    params['lambda']=best_lambda

    print("Training final model:")
    print("Learning rate: ",best_lr)
    print("Max depth: ", best_dep)
    print("Gamma: ", best_gamma)
    print("Lambda: ", best_lambda)

    watchlist = [(dtrain, 'train')]
    num_round = max_steps
    bst = xgb.train(params, dtrain, num_round,watchlist,early_stopping_rounds=5)
    y_pred = bst.predict(xgb.DMatrix(x_test),ntree_limit=bst.best_ntree_limit)
    # Change back to original scale.
    y_pred = np.expm1(y_pred)
    # Save submission
    submission = pd.DataFrame({'Id':list(range(1461,2920)),'SalePrice':y_pred})
    file_name = 'ver_'+str(ver_no)+'_xgb_'+str(best_dep)+'_'+str(best_lr)+'_'+str(best_gamma)+'_'+str(best_lambda)
    submission.to_csv('./submissions/'+file_name+'.csv',index=False)

    #Return Feature Importance-------------------------------------------------------------------------------------------------------------------------
    col_names = dtrain.feature_names
    feature_importance = bst.get_score(importance_type='gain')
    feature_importance = [(item[0],item[1]) for item in feature_importance.items()]
    featureImportance = pd.DataFrame(feature_importance).sort_values(by =1,ascending=False)
    featureImportance = featureImportance.rename(columns = {0:'Feature',1:'Avg_Gain'})

    # Save model and tree structure for later inspections.
    bst.save_model('./tree/'+file_name+'.model')
    bst.dump_model('./tree/'+file_name+'.txt')


    return featureImportance, best_lr,best_dep,best_gamma,best_lambda
