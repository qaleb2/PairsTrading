# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
import seaborn
import seaborn as sns; sns.set(style="whitegrid")

def generate_data(params):
    mu = params[0]
    sigma = params[1]
    return np.random.normal(mu, sigma)

# Set the parameters and the number of datapoints
params = (0, 1)
T = 100

A = pd.Series(index=range(T))
A.name = 'A'

for t in range(T):
    A[t] = generate_data(params)

T = 100

B = pd.Series(index=range(T))
B.name = 'B'

for t in range(T):
    # Now the parameters are dependent on time
    # Specifically, the mean of the series changes over time
    params = (t * 0.1, 1)
    B[t] = generate_data(params)
    
fig, (ax1, ax2) = plt.subplots(nrows =1, ncols =2, figsize=(16,6))

ax1.plot(A)
ax2.plot(B)
ax1.legend(['Series A'])
ax2.legend(['Series B'])
ax1.set_title('Stationary')
ax2.set_title('Non-Stationary')

mean = np.mean(B)

plt.figure(figsize=(12,6))
plt.plot(B)
plt.hlines(mean, 0, len(B), linestyles='dashed', colors = 'r')
plt.xlabel('Time')
plt.xlim([0, 99])
plt.ylabel('Value')
plt.legend(['Series B', 'Mean'])

def stationarity_test(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely stationary.')
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely non-stationary.')
        
stationarity_test(A)
stationarity_test(B)

# Generate daily returns

Xreturns = np.random.normal(0, 1, 100)

# sum up and shift the prices up

X = pd.Series(np.cumsum(
    Xreturns), name='X') + 50
X.plot(figsize=(15,7))

noise = np.random.normal(0, 1, 100)
Y = X + 5 + noise
Y.name = 'Y'

pd.concat([X, Y], axis=1).plot(figsize=(15, 7))

plt.show()

plt.figure(figsize=(12,6))
(Y - X).plot() # Plot the spread
plt.axhline((Y - X).mean(), color='red', linestyle='--') # Add the mean
plt.xlabel('Time')
plt.xlim(0,99)
plt.legend(['Price Spread', 'Mean']);

score, pvalue, _ = coint(X,Y)
print(pvalue)

# Low pvalue means high cointegration!

X_returns = np.random.normal(1, 1, 100)
Y_returns = np.random.normal(2, 1, 100)

X_diverging = pd.Series(np.cumsum(X_returns), name='X')
Y_diverging = pd.Series(np.cumsum(Y_returns), name='Y')


pd.concat([X_diverging, Y_diverging], axis=1).plot(figsize=(12,6));
plt.xlim(0, 99)

print('Correlation: ' + str(X_diverging.corr(Y_diverging)))
score, pvalue, _ = coint(X_diverging,Y_diverging)
print('Cointegration test p-value: ' + str(pvalue))

Y2 = pd.Series(np.random.normal(0, 1, 1000), name='Y2') + 20
Y3 = Y2.copy()

# Y2 = Y2 + 10
Y3[0:100] = 30
Y3[100:200] = 10
Y3[200:300] = 30
Y3[300:400] = 10
Y3[400:500] = 30
Y3[500:600] = 10
Y3[600:700] = 30
Y3[700:800] = 10
Y3[800:900] = 30
Y3[900:1000] = 10


plt.figure(figsize=(12,6))
Y2.plot()
Y3.plot()
plt.ylim([0, 40])
plt.xlim([0, 1000]);

# correlation is nearly zero
print( 'Correlation: ' + str(Y2.corr(Y3)))
score, pvalue, _ = coint(Y2,Y3)
print( 'Cointegration test p-value: ' + str(pvalue))

pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import datetime
import yfinance as yf
yf.pdr_override()

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

############################################################

# DATA


# Gets cleaned data from csv of indices with date colum
df = pd.read_csv('/Users/nori/Desktop/Stocks.csv')

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

# Finding cointegrated pairs
tickers = df.columns
scores, pvalues, pairs = find_cointegrated_pairs(df)
fig, ax = plt.subplots(figsize=(10,10))
seaborn.heatmap(pvalues, xticklabels=tickers, yticklabels=tickers, cmap='RdYlGn_r' 
                , mask = (pvalues >= 0.05)
                )

a=[]
for i in range(len(pairs)):
    S1 = df[pairs[i][0]]
    S2 = df[pairs[i][1]]
    score, pvalue, _ = coint(S1, S2)
    a=a+[pvalue]
    
pairs_min=pairs[a.index(np.min(a))]
print(pairs_min)

# Create stocks1 and stocks2
stock1 = df.iloc[:,93]
stock2 = df.iloc[:,96]





############################################################

# TEST TRADING
# We will not be using this code directly, this is for demonstration only
# We will try to write this code into a class structure

# time_interval = ['2017-01-01','2020-12-31']
# df1 = data.get_data_yahoo(['AMD'],start= time_interval[0],end =time_interval[1])['Close']
# df2 = data.get_data_yahoo(['TTD'],start = time_interval[0], end =time_interval[1])['Close']



# ratio = df1['AMD']/df2['TTD']
# ratio.plot(figsize = (12,6))
# plt.axhline(ratio.mean(),color = 'black')
# plt.xlim('2017-01-01','2020-12-31')
# plt.legend(['Price Ratio'])

ratio = stock1/stock2
ratio.plot(figsize = (12,6))
plt.axhline(ratio.mean(),color = 'black')
plt.legend(['Price Ratio'])



def zscore(series):
    return(series - series.mean()) / np.std(series)



zscore(ratio).plot(figsize=(12,6))
plt.axhline(zscore(ratio).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.show()



print(len(ratio) * .80 ) 
print(len(ratio) * .20 ) 
train = ratio[:767]
test = ratio[767:]




ratios_mavg5 = train.rolling(window=5, center=False).mean()
ratios_mavg30 = train.rolling(window=30, center=False).mean()
std_30 = train.rolling(window=30, center=False).std()
zscore_30_5 = (ratios_mavg5 - ratios_mavg30)/std_30
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg30.index, ratios_mavg30.values)
plt.legend(['Ratio', '5d Ratio MA', '30d Ratio MA'])
plt.ylabel('Ratio')
plt.show()





plt.figure(figsize=(12,6))
zscore_30_5.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()






plt.figure(figsize=(12,6))
train[191:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_30_5>-1] = 0
sell[zscore_30_5<1] = 0
buy[191:].plot(color='g', linestyle='None', marker='^')
sell[191:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratio.min(), ratio.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()





plt.figure(figsize=(12,7))
S1 = stock1.iloc[:767]
S2 = stock2.iloc[:767]
S1[30:].plot(color='b')
S2[30:].plot(color='c')
buyR = 0*S1.copy()
sellR = 0*S1.copy()
# When you buy the ratio, you buy stock S1 and sell S2
buyR[buy!=0] = S1[buy!=0]
sellR[buy!=0] = S2[buy!=0]
# When you sell the ratio, you sell stock S1 and buy S2
buyR[sell!=0] = S2[sell!=0]
sellR[sell!=0] = S1[sell!=0]
buyR[30:].plot(color='g', linestyle='None', marker='^')
sellR[30:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))
plt.legend(['AMD', 'TTD', 'Buy Signal', 'Sell Signal'])
plt.show()



def trade(S1, S2, window1, window2):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0
    
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    
    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            #print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            #print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            #print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
            
    return money


############################################################

# ACTUAL CODE. We need to fill in this class structure


# class PairsTradingAlgorithm(PAlg):
#     def Initialize(self):
#         #Initializing method
#         return 0
    
#     def Trading(self, S1, S2, window1, window2):
#         # ALgorithm that uses sell function, buy function
        
#         return 0
    
#     def Liquidate(self, ):
#         return 0
        


trade(stock1[767:], stock2[767:], 30, 10)



