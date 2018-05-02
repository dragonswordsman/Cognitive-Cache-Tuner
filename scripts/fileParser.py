#!/usr/bin/env python2

def fileParser(path, labelPath, outputDir):
    import numpy as np
    import os

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    features = 0
    data = {}
    i = 0
    num = {}
    for bench in os.listdir(path):
        n = 0
        fileName = path + "/" + bench + "/32k4w64-32k4w64-1.9GHz/" + bench + ".txt"
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
                features = features + 1    
        f.close()
        num[bench] = n
        i = i + 1

    out = []
    for bench in os.listdir(path):
        fileName = path + "/" + bench + "/32k4w64-32k4w64-1.9GHz/" + bench + ".txt"
        f = open(fileName, "r")
        
        X = np.zeros((num[bench], features))
        j = 0
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
            if (j < n):
                X[j, data[split[0]]] = float(split[1])
        
        f.close()
        
        # Get rid of the last data set since it is invalid
        X = X[:-1, :]
        
        Y = []
        fileName = labelPath + "/" + bench + ".labels"
        f = open(fileName, "r")
        for line in f:
            split = line.split()
            Y.append(bench + "_" + split[0])
        
        nn = min(X.shape[0], len(Y))
        X = X[:nn, :]
        Y = Y[:nn]
        out.append([bench, X, Y])
        
    print("Done with file parser")
    return list(data.keys()), out
  