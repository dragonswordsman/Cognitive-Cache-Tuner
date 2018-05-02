import os
import numpy as np
import time
from fileParser import fileParser
start_time = time.time()
    
experimentPath = "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/mcpat"
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
countSimpoints = 0
table = []
mapping = {}
freq = {}
out = open(output + "/labelsToCache.txt", "w")
table.append(["Benchmark", "Simpoint", "Local Label", "CacheConfig", "minEDP"])
print("%-10s %-8s %11s %18s %10s" % ("Benchmark", "Simpoint", "Local Label", "CacheConfig", "minEDP"))
best = {}
for bench in os.listdir(experimentPath):
    simpointDirs = os.listdir(experimentPath + "/" + bench)
    for simpointName in simpointDirs:
        countSimpoints = countSimpoints + 1
#        print("Benchmark " + bench + " Simpoint " + simpointName[9:])
        cacheConfigs = os.listdir(experimentPath + "/" + bench + "/" + simpointName)
        if (len(cacheConfigs) == 0):
            continue
        minEDP = float("inf")
        bestCache = ""
        count = 0
        tot_edp = 0
        for cache in cacheConfigs:
            count = count + 1
            fileDir = experimentPath + "/" + bench + "/" + simpointName + "/" + cache
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
                tot_edp = edp + tot_edp
                freq[bench + "_" + label[bench + "_" + simpointName[9:]][len(bench)+1:] + "_" + cache] = edp
#                print(edp)
                if (edp < minEDP):
                    minEDP = edp
                    bestCache = cache
        best[bench + "_" + simpointName] = minEDP
#                    print(cache)
        mapping[bench + "_" + label[bench + "_" + simpointName[9:]][len(bench)+1:]] = bestCache
        print("%-10s %8s %11s %18s %10f %3d %3d %5.8f" % (bench, simpointName[9:], label[bench + "_" + simpointName[9:]][len(bench)+1:], bestCache, minEDP, count, countSimpoints, tot_edp))
        table.append([bench, simpointName[9:], label[bench + "_" + simpointName[9:]][len(bench)+1:], bestCache, str(minEDP)])
        out.write(label[bench + "_" + simpointName[9:]] + " " + bestCache + "\n")

edp_label = np.array(['Current EDP', 'Base EDP', 'Best EDP', 'Simpoint'])
edp_dict = {}
for bench in os.listdir(experimentPath):
    simpointDirs = os.listdir(experimentPath + "/" + bench)
    for simpointName in simpointDirs:
        countSimpoints = countSimpoints + 1
#        print("Benchmark " + bench + " Simpoint " + simpointName[9:])
        cacheConfigs = os.listdir(experimentPath + "/" + bench + "/" + simpointName)
        if (len(cacheConfigs) == 0):
            continue
        for cache in cacheConfigs:
            current = bench + "_" + label[bench + "_" + simpointName[9:]][len(bench)+1:] + "_" + cache
            base = bench + "_" + label[bench + "_" + simpointName[9:]][len(bench)+1:] + "_" + "32kB4w64-32kB4w64"
#            print(current, freq[current], freq[base], best[bench + "_" + simpointName])
            edp = np.array([freq[current], freq[base], best[bench + "_" + simpointName], label[bench + "_" + simpointName[9:]][len(bench)+1:]])
            edp_dict[bench + "_" + label[bench + "_" + simpointName[9:]][len(bench)+1:]] = edp
            

np.savetxt(output + "/mapping.csv", list(freq.items()), delimiter=',', fmt='%s')

a = np.array(table)
np.savetxt(output + "/table.csv", a, delimiter=",", fmt="%s")
print("Reference table saved to: " + output + "/table.csv")
out.close()        

keys, data = fileParser(path, labelPath, outputDir)  
    
keyss = np.array(keys).reshape((1, len(keys)))
keys = keyss

edp_map = []
Y_cpy = []
for [bench, X, Y] in data:
    Y_cpy.append(list(Y))

    for i in range(len(Y)):
        edp_map.append(edp_dict[Y[i]])
print(Y_cpy)
edp_map = np.array(edp_map)
edp_map = np.vstack((edp_label, edp_map))

for [bench, X, Y] in data:
    count = 0
    for i in range(len(Y)):
        if Y[i] in mapping:
            Y[i] = mapping[Y[i]]
        else:
            Y[i] = "none"
            count = count + 1

dictionary = {}
num = 0
for [bench, X, Y] in data:
    for i in range(len(Y)):
        if Y[i] not in dictionary:
            
            dictionary[Y[i]] = num
            num = num + 1

for [bench, X, Y] in data:
    for i in range(len(Y)):
        Y[i] = dictionary[Y[i]]    


print(dictionary)
#for [bench, X, Y] in data:
#    fileName = outputDir + "/" + bench + ".csv"
#    print(fileName)
#    X = np.array(X)
#    Y = np.array(Y).reshape((X.shape[0], 1))
#    out = np.hstack((X, Y))
#    out = np.vstack((keys, out))
#    np.savetxt(fileName, out, delimiter=',', fmt='%s')

fileName = output + "/master.csv"
label = np.array(["Config"]).reshape((1, 1))
for [bench, X, Y], Y2 in zip(data, Y_cpy):
    edp_map_bench = []
    for i in range(len(Y2)):
        edp_map_bench.append(edp_dict[Y2[i]])
    edp_map_bench = np.array(edp_map_bench)  
    edp_map_bench = np.vstack((edp_label, edp_map_bench))
    print(edp_map_bench.shape)
    print('Creating: ' + outputDir + "/" + bench + ".csv")
    keys = np.vstack((keys, X))
    label = np.vstack((label, np.array(Y).reshape(X.shape[0], 1)))
    np.savetxt(outputDir + "/" + bench + ".csv", np.hstack((np.vstack((np.hstack((keyss, np.array(["Config"]).reshape((1, 1)))), np.hstack((X, np.array(Y).reshape(len(Y), 1))))), edp_map_bench)), delimiter=',', fmt='%s' )

np.savetxt(fileName, np.hstack((keys, label, edp_map)), delimiter=',', fmt='%s')
np.savetxt(output + "/key.csv", np.array([list(dictionary.keys()), list(dictionary.values())]).T, delimiter=',', fmt='%s')
print("--- %s seconds ---" % (time.time() - start_time))




