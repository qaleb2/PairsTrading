#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:26:49 2021

@author: Caleb
"""


from pandas_datareader import data 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import seaborn



############################################################

# DATA


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
np.where(pvalues == pvalues.min())
 

# Stock2
tickers[96] 

# Stock1
tickers[93]

# Create stocks1 and stocks2
stock1 = df.iloc[:,93]
stock2 = df.iloc[:,96]


############################################################

# ACTUAL CODE OOP Pairs Trading Algorithm.




class PairsTradingAlgorithm():
    
    def __init__(self, S1, S2, window1, window2, startamt = 0):
        self.money = startamt
        self.S1 = S1
        self.S2 = S2
        self.window1 = window1
        self.window2 = window2
        
        
        self.ratios = self.S1/self.S2
        self.countS1 = 0
        self.countS2 = 0


    # Buys s2 and sells s1
    def Buy(self, i):
        self.money -= self.S1[i] - self.S2[i] * self.ratios[i]
        self.countS1 += 1
        self.countS2 -= self.ratios[i]
        
        
    # Buys s1 and sells s2
    def Sell(self, i):
        self.money += self.S1[i] - self.S2[i] * self.ratios[i]
        self.countS1 -= 1
        self.countS2 += self.ratios[i]
    
    
    # sells everything
    def Liquidate(self, i):
        self.money += self.S1[i] * self.countS1 + self.S2[i] * self.countS2
        self.countS1 = 0
        self.countS2 = 0
        
        
    def Trade(self):
        
        
        # Deals with special cases
        if (self.window1 == 0) or (self.window2 == 0):
            return 0
        
    
        # Compute rolling mean and rolling standard deviation

        ma1 = self.ratios.rolling(window = self.window1, center=False).mean()
        ma2 = self.ratios.rolling(window = self.window2, center=False).mean()
        std = self.ratios.rolling(window = self.window2, center=False).std()
        zscore = (ma1 - ma2)/std
        
        # zscore = (ratios - ratios.mean() ) / std
        for i in range(len(self.ratios)):
            
            # Sell short if the z-score is > 1
            if zscore[i] < -1:
                self.Sell(i)
                
            # Buy long if the z-score is < -1
            elif zscore[i] > 1:
                self.Buy(i)
            
            # Clear positions if the z-score between -.5 and .5
            elif abs(zscore[i]) < 0.75:
                print(self.countS1, self.countS2)
                self.Liquidate(i)
    
        return self.money
        

# OOP Pairs Trading Algorithm. Can adjust starting amount
thing = PairsTradingAlgorithm(stock1[767:], stock2[767:], 30, 2)
thing.Trade()


# Testing different values for window 2
asdf = []
for i in range(30):
    asdf.append(PairsTradingAlgorithm(stock1[767:], stock2[767:], 30, i).Trade())
    
plt.plot(asdf)

# look at all pairs under 0.05. THink of as a single trading portfolio. Scale by volatility, euqlly weigh them, etc
# Look at more data
# data mining problem. Expanding window calculation. Start with first month, for hedge ratios, etc, then roll forward.
# Do true out of sample testing. 
# Generalize pair to many assets. 
# What if zscore goes to 5. Stop loss. Optimize z scores
# Add value to project by looking at implications of transaction costs. 
# Kalman filter to identify pairs. makes hedge ratios dynamic. 
# Model validation. 

