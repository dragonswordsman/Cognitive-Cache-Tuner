#!/usr/bin/env python2
import numpy as np
import os

path = "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/100b_simpoints"

outputDir = path + "/output/"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
    
benchmark = ["mcf"]

for bench in benchmark:
    features = 0
    n = 0
    data = {}
    fileName = path + "/profiling/" + bench + ".txt"
    f = open(fileName, "r")
    for line in f: 
        line = line[:-1]
        if not line:
            continue
        if "Begin Simulation Statistics" in line:
            n = n + 1
            continue
        if "End Simulation Statistics" in line:
            continue
        
        split = line.split()
        if len(split) > 0 and split[0] not in data:
            data[split[0]] = features
    #        print "%d %s" % (features, split[0])
            features = features + 1    
    f.close()
    
    f = open(fileName, "r")
    j = 0
    X = np.zeros((n, features))
    for line in f: 
        line = line[:-1]
        split = line.split()
    
        if not line:
            continue
        if "Begin Simulation Statistics" in line:
            continue
        if "End Simulation Statistics" in line:
            j = j + 1
            continue
        
        X[j, data[split[0]]] = float(split[1])
    
    f.close()
    
    # Get rid of the last data set since it is invalid
    X = X[:-1, :]
#    print X
    
    Y = []
    fileName = path + "/labels/" + bench + ".labels"
    f = open(fileName, "r")
    for line in f:
        split = line.split()
        Y.append(int(split[0]))
    Y = np.array(Y).reshape((len(Y), 1))
#    print Y
    
    
    n = min(X.shape[0], Y.shape[0])
    X = X[:n, :]
    Y = Y[:n]
    out = np.hstack((X, Y))
    fileName = path + "/output/" + bench + ".csv"
    np.savetxt(fileName, out, delimiter=',')
    
            
print "Done"
  