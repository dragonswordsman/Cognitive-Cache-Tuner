import os
import numpy as np
import time
from fileParser import fileParser
start_time = time.time()
    
experimentPath = "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/experiment"
simpoints =      "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints/simpoints"
output =         "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints"
path =           "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_profile"
labelPath =      "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints/labels";
outputDir =      "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints/output"

label = {}
for simp in os.listdir(simpoints):
#    print(simp[:-10])
    with open(simpoints + "/" + simp, "r") as myfile:
        for line in myfile:
            line = line[:-1].split()
            label[simp[:-10] + "_" + line[0]] = simp[:-10] + "_" + line[1]
#print(label)

table = []
mapping = {}
out = open(output + "/labelsToCache.txt", "w")
table.append(["Benchmark", "Simpoint", "Local Label", "CacheConfig", "minEDP"])
print("%-10s %-8s %11s %18s %10s" % ("Benchmark", "Simpoint", "Local Label", "CacheConfig", "minEDP"))
for bench in os.listdir(experimentPath):
    simpointDirs = os.listdir(experimentPath + "/" + bench)
    for simpointName in simpointDirs:
#        print("Benchmark " + bench + " Simpoint " + simpointName[9:])
        cacheConfigs = os.listdir(experimentPath + "/" + bench + "/" + simpointName)
        if (len(cacheConfigs) == 0):
            continue
        minEDP = float("inf")
        bestCache = ""
        for cache in cacheConfigs:
            fileDir = experimentPath + "/" + bench + "/" + simpointName + "/" + cache;
            with open(fileDir + "/output.txt", "r") as myfile:
                string = filter(None, (myfile.read().split("\n")))
                string = [x.strip(' ') for x in string]
                peakPower = float([x for x in string if "Peak Power" in x][0].split()[3])
                frequency = float([x for x in string if "Core clock Rate(MHz)" in x][0].split()[3])
            with open(fileDir + "/" + bench + ".txt", "r") as myfile:
                string = filter(None, (myfile.read().split("\n")))
                string = [x.strip(' ') for x in string]
                cycles = float([x for x in string if "system.switch_cpus.numCycles" in x][0].split()[1])
#            print (peakPower, frequency, cycles)
            if (peakPower == 0 or frequency == 0 or cycles == 0):
                print("Something doesn't seem right...\n " + str(peakPower) + " " + str(frequency) + " " + str(cycles))
            else:
                runningTime = (1.0 / (frequency * 1000000.0)) * cycles
                edp = peakPower * (runningTime**2)
#                print(edp)
                if (edp < minEDP):
                    minEDP = edp
                    bestCache = cache
#                    print(cache)
        mapping[bench + "_" + label[bench + "_" + simpointName[9:]][len(bench)+1:]] = bestCache
        print("%-10s %8s %11s %18s %10f" % (bench, simpointName[9:], label[bench + "_" + simpointName[9:]][len(bench)+1:], bestCache, minEDP))
        table.append([bench, simpointName[9:], label[bench + "_" + simpointName[9:]][len(bench)+1:], bestCache, str(minEDP)])
        out.write(label[bench + "_" + simpointName[9:]] + " " + bestCache + "\n")

a = np.array(table)
np.savetxt(output + "/table.csv", a, delimiter=",", fmt="%s")
print("Reference table saved to: " + output + "/table.csv")
out.close()        

data = fileParser(path, labelPath, outputDir)  
for [bench, X, Y] in data:
    fileName = outputDir + "/" + bench + ".csv"
    print(fileName)
    for i in range(len(Y)):
        if Y[i] in mapping:
            Y[i] = mapping[Y[i]]
        else:
            Y[i] = "none"
            
    X = np.array(X)
    Y = np.array(Y).reshape((X.shape[0], 1))
    out = np.hstack((X, Y))
    np.savetxt(fileName, out, delimiter=',', fmt='%s')

            
print("--- %s seconds ---" % (time.time() - start_time))




