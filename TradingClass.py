#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:20:30 2021

@author: Caleb
"""

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
                self.Liquidate(i)
                
        return self.money