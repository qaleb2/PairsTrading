#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:26:49 2021

@author: Caleb
"""


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import seaborn
from TradingClass import PairsTradingAlgorithm



############################################################

# DATA PROCESSING 


# Gets cleaned data from csv of indices with date colum
df = pd.read_csv('/Users/Caleb/PairsTrading/Stocks30_524-123.csv')

# Sets date column as index and drops
df = df.set_index("Dates", drop = True)

# Finds out where the NANs Are
is_nan = df.isnull()
row_has_nan = is_nan.any(axis = 1)
rows_with_NaN = df[row_has_nan]

# Gets Rid of Nans
df = df.dropna()

# are there any null values in df
df.isnull().values.any()



# Function to find cointegration
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
    return score_matrix, pvalue_matrix



# Finding cointegrated pairs
tickers = df.columns
scores, pvalues = find_cointegrated_pairs(df)
fig, ax = plt.subplots(figsize=(10,10))
seaborn.heatmap(pvalues, xticklabels=tickers, yticklabels=tickers, cmap='RdYlGn_r' , mask = (pvalues >= 0.05))




pvalues.min()
np.where(pvalues == pvalues.min())

# Too many pairs (roughly half) have cointegration of under 0.05 possibly because they are all nasdaq 100 stocks in same sector

np.where(pvalues < 0.05)
np.where(pvalues < 0.001)
np.union1d(np.where(pvalues < 0.001)[0], np.where(pvalues < 0.001)[1])


# Finding out which stocks were chosen
tickers[49]
tickers[50]


# Actual stocks
stock1 = df.iloc[:,49]
stock2 = df.iloc[:,50]


# creating train test split
split = round(len(stock1)*0.8)
len(stock2)*0.2


totalreturns, tickersreturns = PairsTradingAlgorithm(stock1[:split], (stock1[:split] + stock2[:split]) / 2, 30, 2, 2).Trade()


    

# Finding best window for lowest pvalue stock
profit = []
for i in range(30):
    profit.append(PairsTradingAlgorithm(stock1[:split], (stock1[:split] + stock2[:split]) / 2, 30, i).Trade())
plt.plot(profit)
np.where(profit == max(profit))






# Finding best window for all pairs
bestwindow = []
for j in range(2,30):
    
    # Calculates profit for each pair
    profit = []
    for i in range(len(np.where(pvalues < 0.001)[0])):
        stock1 = df.iloc[:,np.where(pvalues < 0.001)[0][i]]
        stock2 = df.iloc[:,np.where(pvalues < 0.001)[1][i]]
        profit.append(PairsTradingAlgorithm(stock1[:split], stock2[:split], 30, j, 1).Trade()[0])

    plt.plot(profit)
    plt.show()
    bestwindow.append(np.where(profit == max(profit)))

# Finding best window for multiple stocks





############################################
# Testing Zscores (1, 2, 3, 4, 5)

profit=[]
for i in range(len(np.where(pvalues < 0.001)[0])):
    profit.append(PairsTradingAlgorithm(stock1, stock2, 30, 2, 1 + 0.5 * i).Trade()[0])
    print(i)
plt.plot(profit)

max_profit = max(profit)
max_index = profit.index(max_profit)
print(max_index*0.5+0.5)



#############################################

# Model Validation on multiple stocks
profit = []
for i in range(len(np.where(pvalues < 0.001)[0])):
    stock1 = df.iloc[:,np.where(pvalues < 0.001)[0][i]]
    stock2 = df.iloc[:,np.where(pvalues < 0.001)[1][i]]
    profit.append(PairsTradingAlgorithm(stock1[split:], stock2[split:], 30, 2).Trade())
plt.plot(profit)


PairsTradingAlgorithm(stock1[split:], (stock1[split:] + stock2[split:]) / 2, 30, 2).Trade()





# look at all pairs under 0.05. Think of as a single trading portfolio. Scale by volatility, equally weigh them, etc
# Look at more data
# data mining problem. Expanding window calculation. Start with first month, for hedge ratios, etc, then roll forward.
# Do true out of sample testing. 
# Generalize pair to many assets. 
# What if zscore goes to 5. Stop loss. Optimize z scores
# Add value to project by looking at implications of transaction costs. 
# Kalman filter to identify pairs. makes hedge ratios dynamic. 
# Model validation. 

