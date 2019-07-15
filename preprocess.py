# Idea borrowed from this script: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# and https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

import numpy as np
import pandas as pd
from utility import *

def impute_and_add(df_train_or_test,impute_LotFrontage = False,to_cate = False,remove_outlier = False,remove_hard_to_fit = False,linear_model = False):
    '''
    Parameters
    ----------
    df_train_or_test : dataframe, created from the ames housing dataset
               This could be either the train set (with or without y_label) or the test set.
               The train set must contain all the original columns.
               The test set must contain all the original columns except the 'Id' column.
    impute_LotFrontage : boolean, default False
                         If true, impute LotFrontage by median among different neighborhoods.
                         If false, impute LotFrontage by 0.
    to_cate : boolean, default False
                       If true, transform OverallCond and OverallQual to string.
                       If false, keep OverallCond and OverallQual as numeric.
    remove_outlier : boolean, default False
                     If true, remove observations whose GrLivArea are greater than 4000.
                     For the test set, this must be false.
    remove_hard_to_fit : boolean, default False
                         If true, remove selected observations from train set.
                         For the test set, this must be false.
    linear_model : boolean, default False
                   If true, log transform a selected set of columns.
                   Note: for tree models, we don't have to do the log transform.

    Returns
    -------
    df_processed : preprocessed dataframe.
    '''
    # Make a copy so the original dataframe will not be altered.
    df_processed = df_train_or_test.copy()

    if remove_outlier:
        df_processed = df_processed[df_processed.GrLivArea < 4000]

    if remove_hard_to_fit:
        set_of_obs_to_remove = [633,463,1325,31,969,1433,589,1454,496,411,875,729,89,689,971]
        df_processed = df_processed[~df_processed.Id.isin(set_of_obs_to_remove)]

	# PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
    df_processed["PoolQC"] = df_processed["PoolQC"].fillna("None")

	# MiscFeature : data description says NA means "no misc feature"

    df_processed["MiscFeature"] = df_processed["MiscFeature"].fillna("None")

	# Alley : data description says NA means "no alley access"
    df_processed["Alley"] = df_processed["Alley"].fillna("None")

	# Fence : data description says NA means "no fence"
    df_processed["Fence"] = df_processed["Fence"].fillna("None")

	# FireplaceQu : data description says NA means "no fireplace"
    df_processed["FireplaceQu"] = df_processed["FireplaceQu"].fillna("None")

	# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
	# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    # If impute_LotFrontage is false, simply impute by 0.
    if impute_LotFrontage:
        df_processed["LotFrontage"] = df_processed.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    else:
        df_processed['LotFrontage'] = df_processed['LotFrontage'].fillna(0)

	# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        df_processed[col] = df_processed[col].fillna('None')

	# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df_processed[col] = df_processed[col].fillna(0)

	# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        df_processed[col] = df_processed[col].fillna(0)

	# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df_processed[col] = df_processed[col].fillna('None')

	# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
    df_processed["MasVnrType"] = df_processed["MasVnrType"].fillna("None")
    df_processed["MasVnrArea"] = df_processed["MasVnrArea"].fillna(0)

	# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
    df_processed['MSZoning'] = df_processed['MSZoning'].fillna(df_processed['MSZoning'].mode()[0])

	# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
    df_processed.drop(['Utilities'], axis=1,inplace=True)

	# Functional : data description says NA means typical
    df_processed["Functional"] = df_processed["Functional"].fillna("Typ")

	# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    df_processed['Electrical'] = df_processed['Electrical'].fillna(df_processed['Electrical'].mode()[0])

	# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
    df_processed['KitchenQual'] = df_processed['KitchenQual'].fillna(df_processed['KitchenQual'].mode()[0])

	# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    df_processed['Exterior1st'] = df_processed['Exterior1st'].fillna(df_processed['Exterior1st'].mode()[0])
    df_processed['Exterior2nd'] = df_processed['Exterior2nd'].fillna(df_processed['Exterior2nd'].mode()[0])

	# SaleType : Fill in again with most frequent which is "WD"
    df_processed['SaleType'] = df_processed['SaleType'].fillna(df_processed['SaleType'].mode()[0])

	# MSSubClass : Na most likely means No building class. We can replace missing values with None
    df_processed['MSSubClass'] = df_processed['MSSubClass'].fillna("None")

	# Adding total sqfootage feature and weighted TotalSF
    df_processed['TotalSF'] = df_processed['TotalBsmtSF'] + df_processed['1stFlrSF'] + df_processed['2ndFlrSF']
    df_processed['SF_score'] = df_processed['1stFlrSF'] + df_processed['2ndFlrSF'] + 0.75*df_processed['TotalBsmtSF'] + 0.5*df_processed['LowQualFinSF']

    # Encode some categorical features as ordered numbers when there is information in the order
    df_processed = df_processed.replace({"Alley" : {"None":0,"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"None" : 0,"No":1, "Mn" : 2, "Av": 3, "Gd" : 4},
                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2}})

	# add YearsOld and YearSinceRemodel:
    df_processed['YearsOld']  = df_processed['YrSold'] - df_processed['YearBuilt']
    df_processed['YearSinceRemodel'] = df_processed['YrSold'] - df_processed['YearRemodAdd']

	# combine bathroom:
    df_processed['BsmtBath'] = df_processed['BsmtFullBath'] + 0.5* df_processed['BsmtHalfBath']
    df_processed['Bath'] = df_processed['FullBath'] + 0.5 * df_processed['HalfBath']

	# combine basement feature into a binary: whether has basement and a overall score : a weighted sum
    df_processed['Basement'] = df_processed['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)
    t1Pct = (df_processed['BsmtFinSF1']/df_processed['TotalBsmtSF'])
    t1Pct[np.isnan(t1Pct)] = 0
    t2Pct = (df_processed['BsmtFinSF2']/df_processed['TotalBsmtSF'])
    t2Pct[np.isnan(t2Pct)] = 0
    unfPct = (df_processed['BsmtUnfSF']/df_processed['TotalBsmtSF'])
    unfPct[np.isnan(unfPct)] = 0
    logSF = np.log1p(df_processed['TotalBsmtSF'])
    df_processed['BsmtScore'] = 2*df_processed['BsmtQual'] + df_processed['BsmtCond'] + 0.5*df_processed['BsmtExposure'] + df_processed['BsmtFinType1']*t1Pct + df_processed['BsmtFinType2']*t2Pct + unfPct*logSF


    df_processed['add_TotalBaths'] = df_processed['Bath'] + df_processed['BsmtBath']

    df_processed['add_TotRmsAbvGrdWBath'] = df_processed['TotRmsAbvGrd'] + df_processed['add_TotalBaths']
    df_processed['add_SFPerRm'] = round(df_processed['1stFlrSF'] + df_processed['2ndFlrSF']+  df_processed['LowQualFinSF']) / df_processed['add_TotRmsAbvGrdWBath']
    GarSFPerCarPlus1SD = round(df_processed['GarageArea'] / df_processed['GarageCars']).mean() + (1 * round(df_processed['GarageArea'] / df_processed['GarageCars']).std())
    df_processed['add_GarageSpacious'] = round(df_processed['GarageArea'] / df_processed['GarageCars']).apply(lambda x: 1 if x > GarSFPerCarPlus1SD else 0)

# Re: Outdoor space
    df_processed['add_PorchSF'] = df_processed['OpenPorchSF'] + df_processed['EnclosedPorch'] + df_processed['3SsnPorch'] + df_processed['ScreenPorch']
    df_processed['add_OutdoorSF'] = df_processed['LotArea'] - df_processed['1stFlrSF']
    df_processed['add_YardSF'] = df_processed['LotArea'] - df_processed['1stFlrSF'] - df_processed['add_PorchSF'] - df_processed['PoolArea'] - df_processed['WoodDeckSF']

#---Multiply features:
    df_processed["add_OverallGrade"] = df_processed["OverallQual"] * df_processed["OverallCond"]
    df_processed["add_GarageGrade"] = df_processed["GarageQual"] * df_processed["GarageCond"]
    df_processed["add_ExterGrade"] = df_processed["ExterQual"] * df_processed["ExterCond"]
    df_processed["add_KitchenScore"] = df_processed["KitchenAbvGr"] * df_processed["KitchenQual"]
    df_processed["add_FireplaceScore"] = df_processed["Fireplaces"] * df_processed["FireplaceQu"]
    df_processed["add_GarageScore"] = df_processed["GarageArea"] * df_processed["GarageQual"]
    df_processed["add_PoolScore"] = df_processed["PoolArea"] * df_processed["PoolQC"]
    df_processed['add_GrLivArea*OvQual'] = df_processed['GrLivArea'] * df_processed['OverallQual']
    df_processed['add_QualOverall*Exter*Kitch*Bsmt*Garage'] = df_processed['OverallQual'] * df_processed['ExterQual'] * df_processed['KitchenQual'] * df_processed['BsmtQual'] * df_processed['GarageQual']

## Add Binary
# Re: type property/sale
    df_processed['add_NormalSale_bi'] = df_processed['SaleCondition'].apply(lambda x: 1 if x=='Normal' else 0)
    # House completed before sale or not
    df_processed['add_BoughtOffPlan'] = df_processed['SaleCondition'].apply(lambda x: 1 if x=='Partial' else 0)
# Re: Proximity
    df_processed['add_Cond1_2_NearRlrd_bi'] = np.where(df_processed['Condition1'].str[:3] =='RRN', 1, 0) + np.where(df_processed['Condition2'].str[:3] =='RRN', 1, 0)
    df_processed['add_Cond1_2_NearPosFtr_bi'] = np.where(df_processed['Condition1'].str[:3] =='Pos', 1, 0) + np.where(df_processed['Condition2'].str[:3] =='Pos', 1, 0)
    df_processed['add_Cond1_2_NearBusyRd_bi'] = np.where(df_processed['Condition1'].isin(['Artery', 'Feedr']), 1, 0) + np.where(df_processed['Condition1'].isin(['Artery', 'Feedr']), 1, 0)
# Re: Amenities (AC, Paved Drive, Pool, etc.)
    df_processed['add_HasPool_bi'] = df_processed['PoolQC'].apply(lambda x: 0 if x=='None' else 1)
    df_processed['add_HasDeck_bi'] = df_processed['WoodDeckSF'].apply(lambda x: 1 if x>0 else 0)
    df_processed['add_HasPorch_bi'] = df_processed['add_PorchSF'].apply(lambda x: 1 if x>0 else 0)
    df_processed['add_MasVnrStoneBrick_bi'] = np.where(df_processed['MasVnrType'].isin(['Stone', 'BrkFace']), 1, 0)
    df_processed['add_HasFinishedBasement_bi'] = np.where(df_processed['BsmtFinType1'].isin(['GLQ', 'ALQ', 'BLQ', 'Rec']), 1, 0)


    # Log transform some features ( useful in linear models):
    if linear_model:
        Linear_Num_Cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'LotArea', 'GarageArea', 'TotRmsAbvGrd', 'TotalSF', 'BsmtFinSF1']
        df_processed = log_columns(df_processed, Linear_Num_Cols)

    # Transforming some numerical variables that are really categorical
    #MSSubClass=The building class
    df_processed['MSSubClass'] = df_processed['MSSubClass'].apply(str)
    #Changing OverallCond into a categorical variable if required by user
    if to_cate:
        df_processed['OverallCond'] = df_processed['OverallCond'].astype(str)
        df_processed['OverallQual'] = df_processed['OverallQual'].astype(str)
	#Year and month sold are transformed into categorical features.
    df_processed['YrSold'] = df_processed['YrSold'].astype(str)
    df_processed['MoSold'] = df_processed['MoSold'].astype(str)
    df_processed['YrSold-Month'] =  df_processed[['YrSold', 'MoSold']].apply(lambda s: '-'.join(s), axis=1)

    return df_processed
