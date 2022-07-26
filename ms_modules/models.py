# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:40:10 2022

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
from sklearn.decomposition import PCA

def setStartState(trainData, trainLabels, nStart):
        nStart = nStart
        # first get 1 positive and 1 negative point so that both classes are represented and initial classifer could be trained.
        cl1 = np.nonzero(trainLabels==1)[0]
        indices1 = np.random.permutation(cl1)
        indicesKnown = np.array([indices1[0]]);
        cl2 = np.nonzero(trainLabels==0)[0]
        indices2 = np.random.permutation(cl2)
        indicesKnown = np.concatenate(([indicesKnown, np.array([indices2[0]])]));

        return indicesKnown

def RandomForestClassifier_RandomSampling(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num, randInt):
    
    Acc = []
    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    for t in range(ITER_Num):
        logisticRegr.fit(L_X_Set, L_Y_Set)
        Acc.append(logisticRegr.score(X_test, Y_test))
        #print(str(t) +":  "+ str(Acc[t]))

        P = logisticRegr.predict_proba(X_train)  + 1e-7
        index = np.random.randint(len(P))

        L_X_Set = np.vstack((L_X_Set, X_train[index, :]))
        # L_Y_Set = np.vstack((L_Y_Set, Y_train[index]))
        L_Y_Set = np.append(L_Y_Set, [Y_train[index]], axis=0)

        X_train = np.delete(X_train, index, 0)
        Y_train = np.delete(Y_train, index, 0)

    return np.array(Acc)


def RandomForestClassifier_EntropySampling(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num, randInt):
    Acc = []
    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    for t in range(ITER_Num):
        #logisticRegr = LogisticRegression()
        logisticRegr.fit(L_X_Set, L_Y_Set)
        Acc.append(logisticRegr.score(X_test, Y_test))
        #print(str(t) +":  "+ str(Acc[t]))

        P = logisticRegr.predict_proba(X_train)  + 1e-7
        P = -P*np.log(P)
        P = P.sum(axis=1)
        #P1 = np.absolute(logisticRegr.predict_proba(X_train)[:,0]-0.5)
        #index = np.argsort(np.absolute(logisticRegr.predict_proba(X_train)[:,0]-0.5), kind='mergesort')[0]
        index = np.argmin(-P)
        
        # index = np.argsort(np.absolute(logisticRegr.predict_proba(X_train)[:,0]-0.5))[0]
        # print(logisticRegr.predict_proba(X_train)[index,:])
        # print("Data : " + str(X_train[index, :]))

        L_X_Set = np.vstack((L_X_Set, X_train[index, :]))
        # L_Y_Set = np.vstack((L_Y_Set, Y_train[index]))
        L_Y_Set = np.append(L_Y_Set, [Y_train[index]], axis=0)

        X_train = np.delete(X_train, index, 0)
        Y_train = np.delete(Y_train, index, 0)

    return np.array(Acc)

def RandomForestClassifier_CSDivEntSamplingMinMax(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num, Covariance, Covariance2, randInt):

    # Total_Set_X, Total_Set_Y      #Universe
    # X_train, Y_train              #Unlabeled Set
    # L_X_Set, _Y_Set               #Laballed Set
    # X_test, Y_test                #Test Set
    
    Acc = []

    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)
    DivergenceArr = np.zeros((len(X_train), len(X_train)))
    for i in range(len(X_train)):
        DivergenceArr[i] = multivariate_normal.pdf(
            X_train, X_train[i], cov=Covariance)
    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)

    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    c1 = []
    for h in range(len(X_train)):
        c1.append(np.sum(multivariate_normal.pdf(
            Total_Set_X, X_train[h], cov=Covariance2)))
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)

    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L_Arr = []
    for j in range(len(L_X_Set)):
        A_L_Arr.append(np.sum(multivariate_normal.pdf(
            L_X_Set, L_X_Set[j], cov=Covariance)))
    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L = sum(A_L_Arr)

    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L_Arr = []
    for h in range(len(L_X_Set)):
        B_L_Arr.append(np.sum(multivariate_normal.pdf(
            Total_Set_X, L_X_Set[h], cov=Covariance2)))
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L = sum(B_L_Arr)


    cnew1 = []

    for o in range(len(X_train)):
        cnew1.append(np.sum(multivariate_normal.pdf(
            L_X_Set, X_train[o], cov=Covariance)))

    Cons_GaussPeak_cov1 = multivariate_normal.pdf(
        L_X_Set[0], L_X_Set[0], cov=Covariance)

    for t in range(ITER_Num):
        # train model
        #logisticRegr = LogisticRegression()
        logisticRegr.fit(L_X_Set, L_Y_Set)
        # train model

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
    return np.array(Acc)


def RandomForestClassifier_CSDivEntSamplingBeta(Total_Set_X, Total_Set_Y, X_train, L_X_Set, Y_train, L_Y_Set, X_test, Y_test, ITER_Num, Covariance, Covariance2, beta, randInt):

    # Total_Set_X, Total_Set_Y      #Universe
    # X_train, Y_train              #Unlabeled Set
    # L_X_Set, _Y_Set               #Laballed Set
    # X_test, Y_test                #Test Set
    
    Acc = []

    logisticRegr =  RandomForestClassifier(50, n_jobs=8, random_state= randInt)

    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)
    DivergenceArr = np.zeros((len(X_train), len(X_train)))
    for i in range(len(X_train)):
        DivergenceArr[i] = multivariate_normal.pdf(
            X_train, X_train[i], cov=Covariance)
    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)

    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    c1 = []
    for h in range(len(X_train)):
        c1.append(np.sum(multivariate_normal.pdf(
            Total_Set_X, X_train[h], cov=Covariance2)))
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)

    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L_Arr = []
    for j in range(len(L_X_Set)):
        A_L_Arr.append(np.sum(multivariate_normal.pdf(
            L_X_Set, L_X_Set[j], cov=Covariance)))
    # Find All Kernel Values between L_X_Set(L) and Total_Set_X(D)
    A_L = sum(A_L_Arr)

    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L_Arr = []
    for h in range(len(L_X_Set)):
        B_L_Arr.append(np.sum(multivariate_normal.pdf(
            Total_Set_X, L_X_Set[h], cov=Covariance2)))
    # Find All Kernel Values between Total_Set_X(D) and Total_Set_X(D)
    B_L = sum(B_L_Arr)


    cnew1 = []

    for o in range(len(X_train)):
        cnew1.append(np.sum(multivariate_normal.pdf(
            L_X_Set, X_train[o], cov=Covariance)))

    Cons_GaussPeak_cov1 = multivariate_normal.pdf(
        L_X_Set[0], L_X_Set[0], cov=Covariance)

    for t in range(ITER_Num):
        # train model
        #logisticRegr = LogisticRegression()
        logisticRegr.fit(L_X_Set, L_Y_Set)
        # train model

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
    return np.array(Acc)


def Read2x2(filename_train, filename_test):

    dt = np.load(filename_train)
    trainData = dt['x']
    Y_train = dt['y']
            
    scaler = preprocessing.StandardScaler().fit(trainData)
    X_train = scaler.transform(trainData)
    
    dt = np.load(filename_test)
    testData = dt['x']
    Y_test = dt['y']
    X_test = scaler.transform(testData)
    
    X = np.concatenate((X_test, X_train), axis= 0)
    
    cov = np.cov(np.transpose(X),bias=True)
    
    dim = 2

    cov3 = cov*(((4/len(X))*(1/(dim+2)))**(2/(dim+4)))+ cov*(((4/102)*(1/(dim+2)))**(2/(dim+4)))

    cov5 = cov*(((4/102)*(1/(dim+2)))**(2/(dim+4)))*2
    
    return X_train, X_test, Y_train.ravel(), Y_test.ravel(), cov3, cov5

def Read4x4(filename_train, filename_test):
    
    dt = np.load(filename_train)
    trainData = dt['x']
    Y_train = dt['y']
            
    scaler = preprocessing.StandardScaler().fit(trainData)
    X_train = scaler.transform(trainData)

    dt = np.load(filename_test)
    testData = dt['x']
    Y_test = dt['y']
    X_test = scaler.transform(testData)
    
    X = np.concatenate((X_test, X_train), axis= 0)
    
    cov = np.cov(np.transpose(X),bias=True)
    
    dim = 2

    cov3 = cov*(((4/len(X))*(1/(dim+2)))**(2/(dim+4)))+ cov*(((4/202)*(1/(dim+2)))**(2/(dim+4)))

    cov5 = cov*(((4/202)*(1/(dim+2)))**(2/(dim+4)))*2
    
    return X_train, X_test, Y_train.ravel(), Y_test.ravel(), cov3, cov5

def ReadRotated(filename_train, filename_test):
    dt = np.load(filename_train)
    trainData = dt['x']
    Y_train = dt['y']
            
    scaler = preprocessing.StandardScaler().fit(trainData)
    X_train = scaler.transform(trainData)
    
    dt = np.load(filename_test)
    testData = dt['x']
    Y_test = dt['y']
    X_test = scaler.transform(testData)
    
    X = np.concatenate((X_test, X_train), axis= 0)
    
    cov = np.cov(np.transpose(X),bias=True)
    
    dim = 2

    cov3 = cov*(((4/len(X))*(1/(dim+2)))**(2/(dim+4)))+ cov*(((4/202)*(1/(dim+2)))**(2/(dim+4)))

    cov5 = cov*(((4/202)*(1/(dim+2)))**(2/(dim+4)))*2
    
    return X_train, X_test, Y_train.ravel(), Y_test.ravel(), cov3, cov5

def ReadStarium(filename_train_feat, filename_train_labels, filename_test_feat, filename_test_labels):

    dt = sio.loadmat(filename_train_feat)
    trainData = dt['features']
    
    dt = sio.loadmat(filename_train_labels)
    Y_train = dt['labels']
    Y_train[Y_train == -1] = 0
            
    scaler = preprocessing.StandardScaler().fit(trainData)
    X_train = scaler.transform(trainData)
    
    dt = sio.loadmat(filename_test_feat)
    testData = dt['features']
    dt = sio.loadmat(filename_test_labels)
    Y_test = dt['labels']
    Y_test[Y_test == -1] = 0
    
    X_test = scaler.transform(testData)
    
    X = X_train#np.concatenate((X_test, X_train), axis= 0)
    
    dim = 272
    
    cov = np.cov(np.transpose(X),bias=True)

    cov_B = cov*(((4/len(X))*(1/(dim+2)))**(2/(dim+4)))+ cov*(((4/202)*(1/(dim+2)))**(2/(dim+4)))

    cov_A = cov*(((4/202)*(1/(dim+2)))**(2/(dim+4)))*2
    
    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)
    DivergenceArr_A = np.zeros((len(X_train), len(X_train)))
    for i in range(len(X_train)):
        DivergenceArr_A[i] = multivariate_normal.pdf(
            X_train, X_train[i], cov=cov_A, allow_singular=(True))
    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)
    
    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)
    DivergenceArr_B = np.zeros((len(X_train), len(X_train)))
    for i in range(len(X_train)):
        DivergenceArr_B[i] = multivariate_normal.pdf(
            X_train, X_train[i], cov=cov_B, allow_singular=(True))
    # Find All Kernel Values between X_train(D/L) and Total_Set_X(D)
    
    return X_train, X_test, Y_train.ravel(), Y_test.ravel(), cov_B, cov_A, DivergenceArr_A, DivergenceArr_B

