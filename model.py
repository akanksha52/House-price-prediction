import pandas as pd
import numpy as np
import data_preprocessing


def model_normal(df):    
    x=df.drop('price', axis=1)
    y=df['price']
    
    x=np.c_[np.ones(x.shape[0]), x.values]
    
    theta=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    y_pred=x.dot(theta)
    
    return evaluation(y, y_pred)


def model_gradient(df):
    x=df.drop('price', axis=1)
    y=df['price']
    
    x=np.c_[np.ones(x.shape[0]), x.values]
    
    theta=np.zeros(x.shape[1])
    alpha=0.1
    iterations=10000
    m=len(y)                    
    for _ in range(iterations):
        y_pred=x.dot(theta)  
        error=y_pred-y   
        gradient = (1/m) * x.T.dot(error)
        theta-=alpha * gradient      
    
    return evaluation(y, y_pred)


def evaluation(y_true, y_pred):
    mse=np.mean((y_pred-y_true)**2)
    rmse=np.sqrt(mse)
    r2=1-(np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2))    
    return mse, rmse, r2


if __name__=='__main__':
    df=data_preprocessing.preprocess_data('Housing.csv')
    a, b, c=model_normal(df)
    print("Using normal equations to find the coefficient of linear regression, results are as follow: ")
    print("MSE: ", a)
    print("RMSE: ", b)
    print("R-square:", c)
    print()
    print()
    a, b, c=model_gradient(df)
    print("Using gradient to find the coefficient of linear regression, with 10,000 iterations and learning rate set to 0.1, results are as follow: ")
    print("MSE: ", a)
    print("RMSE: ", b)
    print("R-square:", c)
