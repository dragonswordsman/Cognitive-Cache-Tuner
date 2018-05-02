import os
from shutil import copyfile
import time
start_time = time.time()

stats = "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data"
newStatsFiles = "/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/processed_output"

for bench in os.listdir(stats):
    print(bench)
    simpointDirs = os.listdir(stats + "/" + bench)
    for simpointName in simpointDirs:
        cacheConfigs = os.listdir(stats + "/" + bench + "/" + simpointName)
        for cache in cacheConfigs:
            fileDir = stats + "/" + bench + "/" + simpointName + "/" + cache;
            with open(fileDir + "/" + bench + ".txt", "r") as myfile:
                string = myfile.read()
                if len(string) > 0:
                    index = string.rindex("Begin Simulation Statistics")
#                    print("%-11s %-13s %17s %7s" % (bench, simpointName, cache, str(index)))
                    if not os.path.exists(newStatsFiles):
                        os.makedirs(newStatsFiles)
                    if not os.path.exists(newStatsFiles + "/" + bench):
                        os.makedirs(newStatsFiles + "/" + bench)
                    if not os.path.exists(newStatsFiles + "/" + bench + "/" + simpointName):
                        os.makedirs(newStatsFiles + "/" + bench + "/" + simpointName)
                    if not os.path.exists(newStatsFiles + "/" + bench + "/" + simpointName + "/" + cache):
                        os.makedirs(newStatsFiles + "/" + bench + "/" + simpointName + "/" + cache)
                          
                    copyfile(fileDir + "/config.ini", newStatsFiles + "/" + bench + "/" + simpointName + "/" + cache + "/config.ini")
                    f = open(newStatsFiles + "/" + bench + "/" + simpointName + "/" + cache + "/" + bench + ".txt", "w")
                    f.write(string[index:])
                    f.close()
                else:
                    print("File size is 0 "  + fileDir + "\n")
                        

print("Done! New stats files written to:\n" + newStatsFiles)
print("--- %s seconds ---" % (time.time() - start_time))
