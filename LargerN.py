#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:53:44 2021

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

# Finding which pairs were chosen
integrated = np.union1d(np.where(pvalues < 0.005)[0], np.where(pvalues < 0.005)[1])

# Finding out which stocks were chosen
tickers[integrated]


# Actual stocks
stocks = df.iloc[:,integrated]


# Testing n > 2 for one value for window 2
returns = []
for stock in stocks.columns:
    returns.append(PairsTradingAlgorithm(stocks[stock], np.mean(stocks.transpose()), 30, 4).Trade())
plt.plot(returns)




# Testing n > 2 for multiple values for window 2
earnings = []
for i in range(30):
    returns = []
    for stock in stocks.columns:
        returns.append(PairsTradingAlgorithm(stocks[stock], np.mean(stocks.transpose()), 30, i).Trade())
    earnings.append(returns)
    
earnings = pd.DataFrame(data = earnings)
plt.plot(earnings)

for i in earnings:
    plt.plot(i)
    
