#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:05:20 2019

@author: aymeric
"""
import numpy as np 
from sklearn.metrics import mean_squared_error

class IterEstimator:
    
    def __init__(self,model,forward,backward,n_meteo_ft):
        self.model = model
        self.f = forward
        self.b = backward
        self.m = n_meteo_ft
    
    def fit(self,X,y):
        self.model.fit(X,y)
    
    def predict(self,Xt):
        ypred = np.zeros(shape=(Xt.shape[0],))
        for i in range(self.f):
            ytemp = (self.model).predict(Xt)
            ypred = np.column_stack((ypred,ytemp))
            Xt = np.column_stack((Xt[:,1:self.b+1],ytemp,Xt[:,self.b+1+self.m*i:self.b+1+self.m*i+self.m*(self.b+1)]))
        ypred = ypred[:,1:]
        return ypred
    
    def score(self,yp,yt):
        return np.sqrt(mean_squared_error(yp,yt))