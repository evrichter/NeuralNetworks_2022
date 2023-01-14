import pandas as pd
import numpy as np
import itertools
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

housing = fetch_california_housing()
df_input_housing = pd.DataFrame(data=housing['data'], columns=housing['feature_names']) # input data as dataframe
target_housing = housing['target'] # target data array

################################################################## 2.1 #####################################################################

################################################################## 2.1.1 #####################################################################
# get best subset of size 3

def all_subsets(columns):
    
    combinations = list(itertools.combinations(columns, 3))   # get all possible subsets of size 3 (=56)
    combinations = [list(n) for n in combinations] 
    return combinations


def linear_regression(input_data, target_data):
    x_train = input_data
    y_train = target_data
    lin_regressor = LinearRegression()
    lin_regressor.fit(x_train,y_train)  # fit regression model
    y_predicted = lin_regressor.predict(x_train)  # predict y
    
    result = count_mse(y_predicted)
    
    return result


def count_mse(y_predicted):
 
    summation = 0  # variable to store the summation of differences
    n = len(y_predicted) # finding total number of items in list

    for i in range (0,n): 

        difference = y_predicted[i] - target_housing[i]  # finding the difference between observed and predicted value
        squared_difference = difference**2  # taking square of the differene 
        summation = summation + squared_difference  # taking a sum of all the differences

    mse = summation/n  # dividing summation by total values to obtain average
    
    return mse


def subset_selection_8_3():
    
    subsets = all_subsets(df_input_housing.columns)   

    lowest_mse = 0
    lowest_subset = None

    for subset in subsets: 
        subset_mse = linear_regression(df_input_housing[subset], target_housing) # perform linear regression for all subsets
        
        print("For subset: ", subset, "The Mean Squared Error is: " , subset_mse) # print all mses 
        
        if (subset_mse < lowest_mse or lowest_mse == 0): 
            lowest_mse = subset_mse   # get lowest mse
            lowest_subset = subset # get subset producing lowest mse
            
    print("------------------------")
    print("Subset with the lowest_mse: ", lowest_subset)  # print subset producing lowest mse
    print("Lowest_mse: " , lowest_mse)  # print lowest subset


################################################################## 2.1.2 #####################################################################
# use pca to get best subset of size 3

def pca_8_3():

    scaler = StandardScaler() 
    standardised_data = scaler.fit_transform(df_input_housing) # standardise original data

    pca = PCA(n_components=3) # run PCA into 3 dimensions
    pca_data = pca.fit_transform(standardised_data)
    
    n_pcs= pca.components_.shape[0] # number of components
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)] # get the index of the most important feature on components
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    most_important_features = [feature_names[most_important[i]] for i in range(n_pcs)] # get most important features
    selected_features_mse = linear_regression(df_input_housing[most_important_features], target_housing) # get mse for features selected by PCA
    
    print("Features selected by PCA:", most_important_features)
    print("MSE of features selected by PCA:", selected_features_mse)


################################################################## 2.2.#####################################################################
# get the best feature

def linear_regression_with_reshape(input_data, target_data):
    
    x = input_data.to_numpy()
    y = target_data
    x_train = x.reshape(-1, 1)
    y_train = y.reshape(-1, 1)
    lin_regressor = LinearRegression()
    lin_regressor.fit(x_train,y_train)  # fit regression model
    y_predicted = lin_regressor.predict(x_train)  # predict y
    
    mse = count_mse(y_predicted)
    
    return mse


def count_mse(y_predicted):
 
    summation = 0  # variable to store the summation of differences
    n = len(y_predicted) # finding total number of items in list

    for i in range (0,n): 

        difference = y_predicted[i] - target_housing[i]  # finding the difference between observed and predicted value
        squared_difference = difference**2  # taking square of the differene 
        summation = summation + squared_difference  # taking a sum of all the differences

    mse = summation/n  # dividing summation by total values to obtain average
    
    return mse


def feature_selection_8_1():
    
    lowest_mse = 0
    lowest_feature = None
   
    for feature in df_input_housing: 
        feature_data = linear_regression_with_reshape(df_input_housing[feature], target_housing) # perform linear regression for all features
                
        if (feature_data < lowest_mse or lowest_mse == 0): 
            lowest_mse = feature_data   # get lowest mse
            lowest_feature = feature # get feature producing lowest mse
                   
    
    print("Feature selected by LR: ", lowest_feature)  # print feature producing lowest mse
    print("MSE of selected feature: ", lowest_mse)  # print lowest mse
    
    # plot prices vs the selected feature
    
    slope, intercept = np.polyfit(df_input_housing['MedInc'], target_housing, 1) # obtain slope and intercept of regression line   
    
    plt.scatter(df_input_housing['MedInc'], target_housing, color = "purple")
    plt.plot(df_input_housing['MedInc'], slope*df_input_housing['MedInc']+intercept)  # add regression line to scatterplot 
    plt.title('Linear regression: selection of one feature')
    plt.ylabel('Housing Prices')
    plt.xlabel('Selected feature: MedInc')
    plt.show()

###############################################################################################################################################

# pca to get the best feature

    scaler = StandardScaler() 
    standardised_data = scaler.fit_transform(df_input_housing) # standardise original data
    df_standardised = pd.DataFrame(data = standardised_data, columns = housing['feature_names'])

    pca = PCA(n_components=1) # run PCA into 1 dimensions
    pca_data = pca.fit_transform(standardised_data)
    
    n_pcs= pca.components_.shape[0] # number of components
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)] # get the index of the most important feature on components
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    most_important_features = [feature_names[most_important[i]] for i in range(n_pcs)] # get the names
    selected_features_mse = linear_regression(df_input_housing[most_important_features], target_housing) # get mse for features selected by PCA
    
    print("Feature selected by PCA:", most_important_features)
    print("MSE of feature selected by PCA:", selected_features_mse)
    
    # plot prices vs first PC
    flattened_pca = pca_data.flatten()
    slope, intercept = np.polyfit(flattened_pca, target_housing, 1) # obtain slope and intercept of regression line       
    plt.scatter(pca_data, target_housing, color = "purple")
    plt.plot(flattened_pca, slope*flattened_pca+intercept)  # add regression line to scatterplot 
    plt.title('PCA: selection of one PC')
    plt.ylabel('Housing Prices')
    plt.xlabel('Principal Component 1')
    plt.show()

