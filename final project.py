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
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs



# Finding cointegrated pairs
tickers = df.columns
scores, pvalues, pairs = find_cointegrated_pairs(df)
fig, ax = plt.subplots(figsize=(10,10))
seaborn.heatmap(pvalues, xticklabels=tickers, yticklabels=tickers, cmap='RdYlGn_r' , mask = (pvalues >= 0.05))

# Finding which pairs
pvalues.min()
# np.where(pvalues == pvalues.min())
np.where(pvalues < 0.05)[0]
# integrated = np.union1d(np.where(pvalues < 0.05)[0], np.where(pvalues < 0.05)[1])


# Which stocks were chosen
tickers[93]
tickers[96}


# Create log returns for stock1 and stock2
stock1 = ( np.log(df.iloc[:,93]) - np.log(df.iloc[:,93]).shift(1) )[1:]
stock2 = ( np.log(df.iloc[:,96]) - np.log(df.iloc[:,96]).shift(1) )[1:]

# stock1 = df.iloc[:,93]
# stock2 = df.iloc[:,96]


############################################################

# Actual Trading



# OOP Pairs Trading Algorithm. Can adjust starting amount
thing = PairsTradingAlgorithm(stock1[767:], stock2[767:], 30, 2)
thing.Trade()


# Testing different values for window 2
asdf = []
for i in range(30):
    asdf.append(PairsTradingAlgorithm(stock1[767:], stock2[767:], 30, i).Trade())
plt.plot(asdf)


# look at all pairs under 0.05. THink of as a single trading portfolio. Scale by volatility, equally weigh them, etc
# Look at more data
# data mining problem. Expanding window calculation. Start with first month, for hedge ratios, etc, then roll forward.
# Do true out of sample testing. 
# Generalize pair to many assets. 
# What if zscore goes to 5. Stop loss. Optimize z scores
# Add value to project by looking at implications of transaction costs. 
# Kalman filter to identify pairs. makes hedge ratios dynamic. 
# Model validation. 

