#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import csv
import operator

fileName = "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints/output/master.csv"

def readFile(fileName):
    X = []
    with open(fileName) as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
            X.append(row)
    X = np.array(X)
    Y = X[:, -1]
    X = X[:, :-1]
    return X, Y


X, Y = readFile(fileName)
Y = Y[1:].astype(np.float)

frequency = {}
for i in Y:
    if i not in frequency:
        frequency[i] = 1
    else:

        frequency[i] = frequency[i] + 1
        
f = sorted(frequency.items(), key=operator.itemgetter(1))
for i in f:
    print("%-3d %4d" % (i[0], i[1]))