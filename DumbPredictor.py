#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:50:38 2018

@author: aymeric
"""
import numpy as np

#premier prédicteur qui prédit une seule valeur partout, valeur choisie à l'avance
class NullPredictor:
    def __init__(self, value):
        self.value = value 
    
    def fit(self, dataset, labels):
        try:
            self.dimension = labels.shape[1]
        except:
            self.dimension = 1
            
    def predict(self, X):
        return self.value*np.ones((len(X),self.dimension))
    
    def score(self, X, y):
        ypred = self.predict(X)
        return np.mean(ypred == y)

#deuxième prédicteur qui prédit la moyenne des valeurs jusqu'à un certain indice
class MeanPredictor:
    def __init__(self,index):
        self.index = index
    
    def fit(self, dataset, labels):
        try:
            self.dimension = np.array(labels).shape[1]
        except:
            self.dimension = 1
        
    def predict(self, X):
        try:
            m = np.mean(X[:,:self.index+1],axis=1)
            res = m
            for i in range (self.dimension-1):
                res = np.column_stack(res,m)
            return res
        except:
            raise ValueError('Something went wrong. Fit before predict')
    
    def score(self, X, y):
        ypred = self.predict(X)
        return np.mean(ypred == y)

#troisieme predicteur qui predit la valeur indiquée à l'indice donné
class PreviousPredictor:
    def __init__(self,index):
        self.index = index
        
    def fit(self, dataset, labels):
        try:
            self.dimension = np.array(labels).shape[1]
        except: 
            self.dimension = 1
        
    def predict(self, X):
        res = X[:,self.index]
        for i in range(self.dimension-1):
            res = np.column_stack((res,X[:,self.index]))
        return res
    
    def score(self, X, y):
        ypred = self.predict(X)
        return np.mean(ypred == y)