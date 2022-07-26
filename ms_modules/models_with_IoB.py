# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:21:19 2022

@author: m84238180
"""

import warnings
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import decomposition
import math
import random
import struct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
from sklearn import metrics
import time

def LogisticRegression_RandomSampling(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num,randInt):
    
    Acc = []
    IoB = []
    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    for t in range(ITER_Num):
        logisticRegr.fit(L_X_Set, L_Y_Set)
        
        m = metrics.confusion_matrix(logisticRegr.predict(X_test),Y_test)
        
        IoB.append(m[1,1]/(m[1,1] + m[0,1] + m[1,0])) # TP/TP+FP+FN
        
        Acc.append(logisticRegr.score(X_test, Y_test))
        #print(str(t) +":  "+ str(Acc[t]))

        P = logisticRegr.predict_proba(X_train)  + 1e-7
        index = np.random.randint(len(P))

        L_X_Set = np.vstack((L_X_Set, X_train[index, :]))
        # L_Y_Set = np.vstack((L_Y_Set, Y_train[index]))
        L_Y_Set = np.append(L_Y_Set, [Y_train[index]], axis=0)

        X_train = np.delete(X_train, index, 0)
        Y_train = np.delete(Y_train, index, 0)

    return np.array(Acc), np.array(IoB)


def LogisticRegression_EntropySampling(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num,randInt):
    Acc = []
    IoB = []
    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    for t in range(ITER_Num):
        #logisticRegr = LogisticRegression()
        logisticRegr.fit(L_X_Set, L_Y_Set)
        
        m = metrics.confusion_matrix(logisticRegr.predict(X_test),Y_test)
        
        IoB.append(m[1,1]/(m[1,1] + m[0,1] + m[1,0])) # TP/TP+FP+FN
        
        
        Acc.append(logisticRegr.score(X_test, Y_test))
        #print(str(t) +":  "+ str(Acc[t]))

        P = logisticRegr.predict_proba(X_train)  + 1e-7
        P = -P*np.log(P)
        P = P.sum(axis=1)
        #P1 = np.absolute(logisticRegr.predict_proba(X_train)[:,0]-0.5)
        #index = np.argsort(np.absolute(logisticRegr.predict_proba(X_train)[:,0]-0.5))[0]
        index = np.argsort(-P, kind='mergesort')[0]
        #index = np.argmax(P)#np.random.choice(np.flatnonzero(P == P.max()))#np.argmax(P)

        L_X_Set = np.vstack((L_X_Set, X_train[index, :]))
        # L_Y_Set = np.vstack((L_Y_Set, Y_train[index]))
        L_Y_Set = np.append(L_Y_Set, [Y_train[index]], axis=0)

        X_train = np.delete(X_train, index, 0)
        Y_train = np.delete(Y_train, index, 0)

    return np.array(Acc), np.array(IoB)

def LogisticRegression_EnesSamplingMinMax(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num, Covariance, Covariance2, beta, DivArr, DivArr2, indicesKnown,randInt):

    # Total_Set_X, Total_Set_Y      #Universe
    # X_train, Y_train              #Unlabeled Set
    # L_X_Set, _Y_Set               #Laballed Set
    # X_test, Y_test                #Test Set
    
    Acc = []
    IoB = []

    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)
    DivergenceArr = np.delete(DivArr, indicesKnown, 0)
    DivergenceArr = np.delete(DivergenceArr, indicesKnown, 1)


    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    c1 = np.sum(np.delete(DivArr2, indicesKnown, 1), axis = 0).tolist()
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)

    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L_Arr = []
    for j in range(len(L_X_Set)):
        A_L_Arr.append(np.sum(multivariate_normal.pdf(
            L_X_Set, L_X_Set[j], cov=Covariance, allow_singular=(True))))
    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L = sum(A_L_Arr)

    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L_Arr = []
    for h in range(len(L_X_Set)):
        B_L_Arr.append(np.sum(multivariate_normal.pdf(
            Total_Set_X, L_X_Set[h], cov=Covariance2, allow_singular=(True))))
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L = sum(B_L_Arr)


    cnew1 = np.sum(np.delete(DivArr[indicesKnown], indicesKnown, 1), axis = 0).tolist()

    Cons_GaussPeak_cov1 = multivariate_normal.pdf(
        L_X_Set[0], L_X_Set[0], cov=Covariance, allow_singular=(True))

    for t in range(ITER_Num):
        print(t)
        # train model
        #logisticRegr = LogisticRegression()
        logisticRegr.fit(L_X_Set, L_Y_Set)
        # train model

        m = metrics.confusion_matrix(logisticRegr.predict(X_test),Y_test)

        IoB.append(m[1,1]/(m[1,1] + m[0,1] + m[1,0])) # TP/TP+FP+FN

        # test model
        Acc.append(logisticRegr.score(X_test, Y_test))
        # test model

        # Find models entropy about remaining training points
        P = logisticRegr.predict_proba(X_train) + 1e-7
        P = -P*np.log(P)
        P = P.sum(axis=1)
        # Find models entropy about remaining training point

        # Map entropy array according to its min and max
        P_Mapped = (P - min(P))/(max(P) - min(P))
        P = P_Mapped
        # Map entropy array according to its min and max

        # Find Divergence component of opt for each remaining training point (DivergenceArr size is (len(X_train),len(Total_Set_X)) )
        #Divergence2 = (np.sum((c1*np.log((c2 + DivergenceArr))),axis=1))
        Divergence2 = np.log((B_L + c1)) - np.multiply(0.5,np.log(A_L+ np.multiply(2,cnew1) + Cons_GaussPeak_cov1))
        # Find Divergence component of opt for each remaining training point (DivergenceArr size is (len(X_train),len(Total_Set_X)) )

        # Map Divergence array according to its min and max
        if((max(Divergence2) - min(Divergence2)) == 0):
            Divergence = (Divergence2 - min(Divergence2) + 1) / \
                (max(Divergence2) - min(Divergence2) + 1)
        else:
            Divergence = (Divergence2 - min(Divergence2)) / \
                (max(Divergence2) - min(Divergence2))
            
        # Map Divergence array according to its min and max

        # Multiply entropy array and divergence array
        P_Div = np.multiply(P, Divergence)
        # Multiply entropy array and divergence array

        # Find its max
        index = np.random.choice(np.flatnonzero(P_Div == P_Div.max()))#np.argmax(P_Div)
        # Find its max

        # Add maximizer point to library
        L_X_Set = np.vstack((L_X_Set, X_train[index, :]))
        L_Y_Set = np.append(L_Y_Set, [Y_train[index]], axis=0)
        # Add maximizer point to library

        # Update c2
        cnew1 = cnew1 + DivergenceArr[index]
        # Update c2

        # update A_L B_L
        A_L = A_L + np.multiply(2,cnew1[index]) + Cons_GaussPeak_cov1
        B_L = B_L + c1[index]
        # update A_L B_L

        # Delete maximizer point from unlabeled set
        X_train = np.delete(X_train, index, 0)
        Y_train = np.delete(Y_train, index, 0)
        # Delete maximizer point from unlabeled set

        cnew1 = np.delete(cnew1, index, 0)
        c1 = np.delete(c1, index, 0)

        # Delete maximizer point from DivergenceArr
        DivergenceArr = np.delete(DivergenceArr, index, 0)
        DivergenceArr = np.delete(DivergenceArr, index, 1)
        # Delete maximizer point from unlabeled set
    return np.array(Acc), np.array(IoB)


def LogisticRegression_EnesSampling1(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num, Covariance, Covariance2, beta, DivArr, DivArr2, indicesKnown,randInt):

    # Total_Set_X, Total_Set_Y      #Universe
    # X_train, Y_train              #Unlabeled Set
    # L_X_Set, _Y_Set               #Laballed Set
    # X_test, Y_test                #Test Set
    
    # Covariance=cov_A
    # Covariance2=cov_B
    # DivArr= DivArr_A
    # DivArr2=DivArr_B
    
    Acc = []
    IoB = []

    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    # Find All Kernel Values between X_train(D/L) and X_train(D/L)
    DivergenceArr = np.delete(DivArr, indicesKnown, 0)
    DivergenceArr = np.delete(DivergenceArr, indicesKnown, 1)


    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    c1 = np.sum(np.delete(DivArr2, indicesKnown, 1), axis = 0).tolist() #delete known columns sum of each elements of each column
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)

    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L_Arr = []
    for j in range(len(L_X_Set)):
        A_L_Arr.append(np.sum(multivariate_normal.pdf(
            L_X_Set, L_X_Set[j], cov=Covariance, allow_singular=(True))))
    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L = sum(A_L_Arr)

    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L_Arr = []
    for h in range(len(L_X_Set)):
        B_L_Arr.append(np.sum(multivariate_normal.pdf(
            Total_Set_X, L_X_Set[h], cov=Covariance2, allow_singular=(True))))
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L = sum(B_L_Arr)


    cnew1 = np.sum(np.delete(DivArr[indicesKnown], indicesKnown, 1), axis = 0).tolist()

    Cons_GaussPeak_cov1 = multivariate_normal.pdf(
        L_X_Set[0], L_X_Set[0], cov=Covariance, allow_singular=(True))

    for t in range(ITER_Num):
        #import pdb; pdb.set_trace()
        print(t)
        # train model
        #logisticRegr = LogisticRegression()
        logisticRegr.fit(L_X_Set, L_Y_Set)
        # train model

        m = metrics.confusion_matrix(logisticRegr.predict(X_test),Y_test)

        IoB.append(m[1,1]/(m[1,1] + m[0,1] + m[1,0])) # TP/TP+FP+FN

        # test model
        Acc.append(logisticRegr.score(X_test, Y_test))
        # test model

        # Find models entropy about remaining training points
        P = logisticRegr.predict_proba(X_train) + 1e-7
        P = -P*np.log(P)
        P = P.sum(axis=1)
        # Find models entropy about remaining training point

        # Map entropy array according to its min and max
        P_Mapped = P#(P - min(P))/(max(P) - min(P))
        P = P_Mapped
        # Map entropy array according to its min and max

        # Find Divergence component of opt for each remaining training point (DivergenceArr size is (len(X_train),len(Total_Set_X)) )
        #Divergence2 = (np.sum((c1*np.log((c2 + DivergenceArr))),axis=1))
        Divergence2 = np.log((B_L + c1)) - np.multiply(0.5,np.log(A_L+ np.multiply(2,cnew1) + Cons_GaussPeak_cov1))
        # Find Divergence component of opt for each remaining training point (DivergenceArr size is (len(X_train),len(Total_Set_X)) )

        # Map Divergence array according to its min and max
        Divergence = np.power(np.e, np.multiply(beta,Divergence2))#(Divergence2 - min(Divergence2)) / \
            #(max(Divergence2) - min(Divergence2))
        # Map Divergence array according to its min and max

        # Multiply entropy array and divergence array
        P_Div = np.multiply(P, Divergence)
        # Multiply entropy array and divergence array

        # Find its max
        index = np.random.choice(np.flatnonzero(P_Div == P_Div.max()))#np.argmax(P_Div)
        # Find its max

        # Add maximizer point to library
        L_X_Set = np.vstack((L_X_Set, X_train[index, :]))
        L_Y_Set = np.append(L_Y_Set, [Y_train[index]], axis=0)
        # Add maximizer point to library

        # Update c2
        cnew1 = cnew1 + DivergenceArr[index]
        # Update c2

        # update A_L B_L
        A_L = A_L + np.multiply(2,cnew1[index]) + Cons_GaussPeak_cov1
        B_L = B_L + c1[index]
        # update A_L B_L

        # Delete maximizer point from unlabeled set
        X_train = np.delete(X_train, index, 0)
        Y_train = np.delete(Y_train, index, 0)
        # Delete maximizer point from unlabeled set

        cnew1 = np.delete(cnew1, index, 0)
        c1 = np.delete(c1, index, 0)

        # Delete maximizer point from DivergenceArr
        DivergenceArr = np.delete(DivergenceArr, index, 0)
        DivergenceArr = np.delete(DivergenceArr, index, 1)
        # Delete maximizer point from unlabeled set
    return np.array(Acc), np.array(IoB)

