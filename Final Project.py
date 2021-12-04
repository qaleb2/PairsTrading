#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:26:49 2021

@author: Caleb
"""


import numpy as np
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



# Too many pairs (roughly half) have cointegration of under 0.05 possibly because they are all nasdaq 100 stocks in same sector
# Finding which pairs are cointegrated
pvalues.min()
np.where(pvalues < 0.05)
np.union1d(np.where(pvalues < 0.001)[0], np.where(pvalues < 0.001)[1])


# Finding out which stocks were chosen
tickers[49]
tickers[50]

# Actual stocks
stock1 = df.iloc[:,49]
stock2 = df.iloc[:,50]


# creating train test split
len(stock1)*0.8
len(stock2)*0.2

# Testing n = 2 for one value for window 2
thing = PairsTradingAlgorithm(stock1[:1391], stock2[:1391], 30, 2)
thing.Trade()


# Testing for multiple stocks
for i in range(len(np.where(pvalues < 0.001)[0])):
    print(i)
    stock1 = df.iloc[:,i]
    stock2 = df.iloc[:,i]
    PairsTradingAlgorithm(stock1[:1391], stock2[:1391], 30, 2)
    


# Testing different values for window 2
profit = []
for i in range(30):
    profit.append(PairsTradingAlgorithm(stock1[:1391], (stock1[:1391] + stock2[:1391]) / 2, 30, i).Trade())
plt.plot(profit)

np.where(profit == max(profit))


#############################################
# Model Validation

PairsTradingAlgorithm(stock1[1391:], (stock1[1391:] + stock2[1391:]) / 2, 30, 2).Trade()






# look at all pairs under 0.05. THink of as a single trading portfolio. Scale by volatility, equally weigh them, etc
# Look at more data
# data mining problem. Expanding window calculation. Start with first month, for hedge ratios, etc, then roll forward.
# Do true out of sample testing. 
# Generalize pair to many assets. 
# What if zscore goes to 5. Stop loss. Optimize z scores
# Add value to project by looking at implications of transaction costs. 
# Kalman filter to identify pairs. makes hedge ratios dynamic. 
# Model validation. 

