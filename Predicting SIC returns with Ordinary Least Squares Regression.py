# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:24:42 2020

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

sic=pd.read_excel(r'C:\Users\Dell\Downloads\sicALL.xlsx') 
#print(sic.tail())

#create direction with classification funtion
#def classificationConditions(df):
    #df['Opening Price-Closing Price VWAP']=df['Opening Price (GHS)']-\
        #df['Closing Price VWAP (GHS)']
    #df['High-Low']=df['Year High (GHS)']-df['Year Low (GHS)']
    #df=df.dropna()
    #X=df[['Opening Price-Closing Price VWAP','High-Low']]
    #Y=np.where(df['Closing Price VWAP (GHS)'].shift(-1)>\
                         #df['Closing Price VWAP (GHS)'],1,-1)
    #return(X,Y)

#create magnitude variables
def regressionConditions(df):
    df['Opening Price-Closing Price VWAP']=df['Opening Price (GHS)']-\
        df['Closing Price VWAP (GHS)']
    df['High-Low']=df['Year High (GHS)']-df['Year Low (GHS)']
    df=df.dropna()
    X=df['Opening Price-Closing Price VWAP'],df['High-Low']
    Y=df['Closing Price VWAP (GHS)'].shift(-1)-\
      df['Closing Price VWAP (GHS)']
    return(sic,X,Y)


# partitioning ourdataset into two datasets: training and testing.
def createDataSplit(X,Y,splitRatio=0.8):
    return train_test_split(X,Y,shuffle=False,\
           train_size=splitRatio)

#create variables with sic data in regression function        
sic,X,Y=regressionConditions(sic)

#splitting datasets
xTrain,xTest,yTrain,yTest=createDataSplit(X,Y,splitRatio=0.8)
xTest.fillna(xTrain.mean(),inplace=True)
yTest.fillna(yTrain.mean(),inplace=True)


#fit training datasets onto regression module
ols=linear_model.LinearRegression()
ols.fit(xTrain,yTrain)

#print('Coefficients:',ols.coef_)

# The mean squared error(Train Sample)
#print('mean squared error:',mean_squared_error(yTrain,ols.predict(xTrain)))

# Explained variance score: 1 is perfect prediction(Train Sample)
#print('Variance score:',r2_score(yTrain,ols.predict(xTrain)))

# The mean squared error(Test Sample)
#print('mean squared error:',mean_squared_error(yTest,ols.predict(xTest)))

# Explained variance score: 1 is perfect prediction(Test Sample)
#print('Variance score:',r2_score(yTest,ols.predict(xTest)))

sic['predictedSignal']=ols.predict(X)
sic['sicReturns']=np.log(sic['Closing Price VWAP (GHS)']/\
                         sic['Closing Price VWAP (GHS)'].shift(1))

def calculateReturns(df,splitValue,symbol):
    cummulativeReturn=df[splitValue:]['%s_Returns' % symbol].cumsum()*100
    df['strategyReturns']=df['%s_Returns' % symbol]*df['predictedSignal'].shift(1)
    return cummulativeReturn

def strategyReturn(df,splitValue,symbol):
    cummulativeStrategyReturn=df[splitValue:]['strategyReturns'].cumsum()*100
    return cummulativeStrategyReturn

cummulativeSICreturn=calculateReturns(sic,len(xTrain),'SIC')
cummulativeStrategyReturn=strategyReturn(sic,len(xTrain),'SIC')

def plot(cummulativeSymbolReturn,cummulativeStrategyReturn,symbol):
    plt.figure(figsize=(10,5))
    plt.plot(cummulativeSymbolReturn,label='%s Returns' % symbol)
    
plot(cummulativeSICreturn,cummulativeStrategyReturn,'SIC')

def sharpeRatio(symbolReturns,strategyReturns):
    strategySTD=strategyReturns.std()
    sharpe=(strategyReturns-symbolReturns)/strategySTD
    return sharpe.mean()

print(sharpeRatio(cummulativeStrategyReturn,cummulativeSICreturn))

