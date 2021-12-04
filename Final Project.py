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
df = pd.read_csv('/Users/Caleb/Documents/Classes/Boston/Fall 2021/703/Final Project/Stocks.csv')

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



# Finding which pairs

# Testing n = 2
# pvalues.min()
# pair = np.where(pvalues == pvalues.min())

# Testing n > 2
integrated = np.union1d(np.where(pvalues < 0.05)[0], np.where(pvalues < 0.05)[1])




# Finding out which stocks were chosen

# n = 2
# tickers[pair]

# n > 2
tickers[integrated]




# Stock Inputs

# Testing n = 2

# Actual stocks
# stock1 = df.iloc[:,93]
# stock2 = df.iloc[:,96]


# Testing n > 2

# Actual stocks
stocks = df.iloc[:,integrated]



# Testing n = 2 for one window
# thing = PairsTradingAlgorithm(stock1[767:], stock2[767:], 30, 2)
# thing.Trade()

# Testing n > 2 for one window
returns = []
for stock in stocks.columns:
    returns.append(PairsTradingAlgorithm(stocks[stock], np.mean(stocks.transpose()), 30, 2).Trade())
plt.plot(returns)



# Testing n = 2
# Testing different values for window 2

# asdf = []
# for i in range(30):
#     asdf.append(PairsTradingAlgorithm(stock1[767:], (stock1[767:] + stock2[767:]) / 2, 30, i).Trade())
# plt.plot(asdf)


# Testing n > 2
# Testing different values for window 2
earnings = []
for i in range(30):
    returns = []
    for stock in stocks.columns:
        returns.append(PairsTradingAlgorithm(stocks[stock], np.mean(stocks.transpose()), 30, i).Trade())
    earnings.append(returns)
    
for i in earnings:
    plt.plot(i)




# look at all pairs under 0.05. THink of as a single trading portfolio. Scale by volatility, equally weigh them, etc
# Look at more data
# data mining problem. Expanding window calculation. Start with first month, for hedge ratios, etc, then roll forward.
# Do true out of sample testing. 
# Generalize pair to many assets. 
# What if zscore goes to 5. Stop loss. Optimize z scores
# Add value to project by looking at implications of transaction costs. 
# Kalman filter to identify pairs. makes hedge ratios dynamic. 
# Model validation. 

