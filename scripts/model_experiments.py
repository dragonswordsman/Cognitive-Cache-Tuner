import os
import operator
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import time
#from progress.bar import Bar
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

#DATASET_PATH = '/Users/Sam/Desktop/ECE523/Cognitive-Cache-Tuner/data/'
DATASET_PATH =   '/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints/'
BENCHMARK_PATH = '/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints/output/'
#RAWDATA_PATH =   '/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/mcpat/'
LABEL_PATH =     '/Users/diegojimenez/OneDrive/School/Senior/Semester2/ECE 523/Cognitive-Cache-Tuner/data/10b_simpoints/labels/'

def generate_data(filename):
    ''' Loads master_dataset from .csv file and returns a data array and a vector of labels'''
    with open(filename) as f:
        reader = csv.reader(f)
        master = list(reader)
    mask = []
    master = np.array(master)
    edp = master[1:, -4:]
    master = master[:, :-4]
    for i in range(master.shape[1]):
        if "cache" in master[0, i] or "Config" in master[0, i]:
            mask.append(True)
        else:
            mask.append(False)
    
    all_data = np.array(master[1:]).astype(float)
    feature_names =  np.array(master[0]).astype(str)
    
    feature_names = feature_names[:-1]
    data          = np.nan_to_num(all_data[:,:-1])
    labels        = all_data[:,-1]
    labels = thresholdLabels(labels, threshold=10)
    return data, labels, feature_names, edp

def split_data(data, labels, edp):
    '''Splits a single data array into training and testing by randomly selecting indices'''
    train_ind, test_ind = randomize_data_indeces(data.shape[0], percent_training=0.80)
    training = (data[train_ind,:], labels[train_ind])
    testing  = (data[test_ind, :], labels[test_ind])
    edp_     = (edp[train_ind, :], edp[test_ind])
    return training, testing, edp_

def randomize_data_indeces(size, percent_training=0.80):
    '''Generates a exclusive lists of indeces to paritition training and test data '''
    ind = np.array(range(0, size))
    np.random.seed(1)
    np.random.shuffle(ind)
    train_ind = np.array(ind[:int(size*percent_training)])
    test_ind  = np.array(ind[int(size*percent_training):])
    return train_ind, test_ind

def feature_selection_experiment(data, labels, feature_names, k=10):
    var_mask = VarianceThreshold(threshold=0.0).fit(data).get_support()
    data = data[:, var_mask]
    fs_percentile = SelectKBest(k=k)
    fs_percentile.fit(data, labels)
    idx_mask = fs_percentile.get_support()
#    for i in range(len(idx_mask)):
#        if idx_mask[i] == True:
#            print(i)
#    print(feature_names[idx_mask])
    return data[:,idx_mask], var_mask, idx_mask

def AdaBoostExperiment(X_train, Y_train, X_test, Y_test):
    ensemble_size = [2, 5, 10, 25, 50, 100]
    results = []

    print('\n==Begin AdaBoost Experiment=====')
    for ensemble in ensemble_size:
        AdaBoost = AdaBoostClassifier(n_estimators=ensemble)
        AdaBoost.fit(X_train, Y_train)
        score = AdaBoost.score(X_test, Y_test)
        results.append(score)
        print('Num Estimators: %d\tAccuracy: %f' %(ensemble, score))

def SVMExperiment(X_train, Y_train, X_test, Y_test):
    kernels = ['linear', 'poly', 'rbf']
    slack   = [0.01, 0.1, 1.0]
    degree  = [1, 2, 3]
    results = []
    
    print('\n==Begin SVM Experiment=====')
    #bar = Bar('SVM Classifer', max=len(kernels)*len(slack)*len(degree))
    for kernel in kernels:
        for deg in degree:
            for C in slack:
                SupportVectorMachine = SVC(C=C, kernel=kernel, degree=deg)
                SupportVectorMachine.fit(X_train, Y_train)
                accuracy = SupportVectorMachine.score(X_test, Y_test)
                results.append(accuracy)
                print('Kernel: %s\tDegree: %d\tC: %f\tAccuracy: %f' \
                        %(kernel, deg, C, accuracy))
                #bar.next()

def NeuralNetworkExperiment(X_train, Y_train, X_test, Y_test):
    num_labels = int(np.max(np.concatenate((Y_train,Y_test),axis=0))+1)
    Y_train = np_utils.to_categorical(Y_train,num_labels)
    Y_test  = np_utils.to_categorical(Y_test, num_labels)

    network_depth = [1,2]
    num_neurons   = [2, 10, 50, 100, 250]
    network_reg   = [False, True]
    results = []

    print('\n==Begin Neural Network Experiment=====')
    for depth in network_depth:
        for layer_size in num_neurons:
            for reg in network_reg:
                model = Sequential()
                model.add(Dense(layer_size, input_dim=data.shape[1]))
                if reg:
                    model.add(Dropout(0.25))
                if depth == 2:
                    model.add(Dense(layer_size))
                    if reg:
                        model.add(Dropout(0.25))
                model.add(Dense(num_labels, activation='softmax'))

                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
                model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=0)
                score = model.evaluate(X_test, Y_test, verbose=0)
                results.append(score[1])
                print('Depth: %d\tNeurons: %d\tDropout: %r\tAccuracy: %f' \
                        %(depth, layer_size, reg, score[1]))
    return results

def KNNExperiment(X_train, Y_train, X_test, Y_test):
    num_neighbors = [5, 10, 50, 100]
    num_neighbors = range(1, 10)
    results = []

    print('\n==Begin KNN Experiment=====')
    for neighbors in num_neighbors:
        NearestNeighbor = KNeighborsClassifier(n_neighbors=neighbors)
        NearestNeighbor.fit(X_train, Y_train)
        score = NearestNeighbor.score(X_test, Y_test)
        results.append(score)
        print('Num Neighbors: %d\tAccuracy: %f' % (neighbors, score))
    print(max(results))
    return results

def FindBestClassifier(X_train, Y_train, X_test, Y_test):
    names = [
#             "GradientBoostingRegressor",
#             "Nearest Neighbors", 
#             "Decision Tree", 
#             "Random Forest", 
             "Neural Net" 
#             "AdaBoost",
#             "Bagging",
#             "Naive Bayes", 
#             "Linear SVM", 
#             "RBF SVM"
             ]
    
#    classifiers = [
#        GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1),
#        KNeighborsClassifier(3),
#        SVC(kernel="linear", C=0.025),
#        SVC(gamma=2, C=1),
#        DecisionTreeClassifier(max_depth=5),
#        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#        MLPClassifier(alpha=1),
#        AdaBoostClassifier(n_estimators=5),
#        GaussianNB(), 
#        SVC(kernel="linear", C=0.025),
#        SVC(gamma=2, C=1)]
    
    results = []
    for name in names:
        print(name)
        if name == "GradientBoostingRegressor":
            n_estimators = range(10, 100)
            learning_rate = np.arange(0.1, 1.0)
            max_depth = range(1, 10)
            cur_max = 0
            for n in n_estimators:
                for l in learning_rate:
                    for m in max_depth:
                        score = GradientBoostingRegressor(n_estimators=n, 
                                                          learning_rate=l, 
                                                          max_depth=m) \
                        .fit(X_train, Y_train) \
                        .score(X_test, Y_test)
                        if (score > cur_max):
                            cur_max = score
                            print(n, l, m, score)
            results.append(cur_max)
                        
        elif name == "Nearest Neighbors":
            n_neighbors = range(2, 100)
            cur_max = 0
            for n in n_neighbors:
                score = KNeighborsClassifier(n) \
                .fit(X_train, Y_train) \
                .score(X_test, Y_test)
                if (score > cur_max):
                    cur_max = score
                    print(n, score)
            results.append(cur_max)

        elif name == "Decision Tree":
            max_depth = range(1, 70)
            cur_max = 0
            re = []
            for m in max_depth:
                score = DecisionTreeClassifier(max_depth=m) \
                .fit(X_train, Y_train) \
                .score(X_test, Y_test)
                if (score > cur_max):
                    cur_max = score
                    print(m, score)
                re.append(score)
            results.append(cur_max)
            plt.scatter(range(1, 70), re)
            plt.show()
        
        elif name == "Bagging":
            n_estimators = range(2, 100)
            cur_max = 0
            for n in n_estimators:
                score = BaggingClassifier(n_estimators=n) \
                .fit(X_train, Y_train) \
                .score(X_test, Y_test)
                if (score > cur_max):
                    cur_max = score
                    print(n, score)
            results.append(cur_max)
        
        elif name == "Random Forest":
            max_depth = range(1, 20)
            n_estimators = range(1, 30)
            max_features = range(1, 10)
            cur_max = 0
            re = []
            for m1 in max_depth:
                print(m1)
                for n in n_estimators:
                    for m2 in max_features:
                        score = RandomForestClassifier(max_depth=m1, 
                                                       n_estimators=n, 
                                                       max_features=m2) \
                        .fit(X_train, Y_train) \
                        .score(X_test, Y_test)
                        re.append(score)
                        if (score > cur_max):
                            cur_max = score
                            print(m1, n, m2, score)
            results.append(cur_max)

            
        elif name == "Neural Net":
            alpha = range(50, 350, 50)
            activation = ['identity', 'logistic', 'tanh', 'relu']
            learning = ['constant', 'invscaling', 'adaptive']
            solver = ['lbfgs', 'sgd', 'adam']
            cur_max = 0
            for a in alpha:
                print(a)
                for act in activation:
                    for l in learning:
                        for s in solver:
                            score = MLPClassifier(hidden_layer_sizes=[a, a], activation=act, solver=s, learning_rate=l, alpha=0.04) \
                            .fit(X_train, Y_train) \
                            .score(X_test, Y_test)
                            if (score > cur_max):
                                cur_max = score
                                print(a, act, s, l, score)
            results.append(cur_max)

        elif name == "AdaBoost":
            n_estimators = range(2, 100)
            cur_max = 0
            for n in n_estimators:
                score = AdaBoostClassifier(n_estimators=n) \
                .fit(X_train, Y_train) \
                .score(X_test, Y_test)
                if (score > cur_max):
                    cur_max = score
                    print(n, score)
            results.append(cur_max)

        elif name == "Naive Bayes":
            score = GaussianNB() \
            .fit(X_train, Y_train) \
            .score(X_test, Y_test)
            print(score)
            results.append(score)
            
        elif name == "Linear SVM":
            C = np.arange(3.0, 4.0, 0.05)
            cur_max = 0
            for c in C:
                print(c)
                score = SVC(kernel="linear", C=c) \
                .fit(X_train, Y_train) \
                .score(X_test, Y_test)
                if (score > cur_max):
                    cur_max = score
                    print(c, score)
            results.append(cur_max)
            
        elif name == "RBF SVM":
            C = np.arange(0.5, 4.0, 0.5)
            gamma = range(1, 10)
            cur_max = 0
            for c in C:
                print(c)
                for g in gamma:
                    score = SVC(gamma=g, C=c) \
                    .fit(X_train, Y_train) \
                    .score(X_test, Y_test)
                    if (score > cur_max):
                        cur_max = score
                        print(c, g, score)
            results.append(cur_max)
        else:
            print("Not Found")
            
        print(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(X_train, Y_train).score(X_test, Y_test))
#        classifier.fit(X_train, Y_train)
#        score = classifier.score(X_test, Y_test)
#        results.append(score)
#        print("%-28s %5.5f" % (name, score))
    print(results)


def plotEDPComparison(X_train, X_test, edp_train, edp_test, var_mask, idx_mask, clf):
    x_train = np.zeros((0, X_train.shape[1]))
    y_train = np.array([])
    x_test = np.zeros((0, X_test.shape[1]))
    y_test = np.array([])
    edp_train = np.zeros((0, edp_train.shape[1]))
    edp_test = np.zeros((0, edp_test.shape[1]))
    
    bench_data = []
    for bench in os.listdir(BENCHMARK_PATH):
        d, l, f, e = generate_data(BENCHMARK_PATH + bench)

        d = d[:, var_mask]
        d = d[:, idx_mask]
        (x_train1, y_train1), (x_test1, y_test1), (edp_train1, edp_test1) = split_data(d, l, e)
        
        bench_data.append([bench.replace('.csv', ''), x_test1, y_test1, edp_test1])
#        d1, l1 = SMOTE().fit_sample(d, l)
#        (x_train1, y_train1), (X_test2, Y_test2), (asdf1, asdf2) = split_data(d1, l1)

        x_train = np.vstack((x_train, x_train1))
        y_train = np.concatenate((y_train, y_train1))
        x_test = np.vstack((x_test, x_test1))
        y_test = np.concatenate((y_test, y_test1))
        edp_test = np.vstack((edp_test, edp_test1))

    cache_map = {}
    with open(DATASET_PATH+'key.csv') as f:
        reader = csv.reader(f)
        master = list(reader)
    master = np.array(master)
    for i in range(len(master)):
        cache_map[int(master[i, 1])] = master[i, 0]
        
    edp_map = {}
    with open(DATASET_PATH+'mapping.csv') as f:
        reader = csv.reader(f)
        master = list(reader)
    master = np.array(master)
    for i in range(len(master)):
        edp_map[master[i, 0]] = float(master[i, 1])


    print(x_train.shape)

#    x_train, y_train = SMOTE().fit_sample(x_train, y_train)

    print(x_train.shape)
    
    rfc = clf.fit(x_train, y_train)
    bar_bench = []
    bar_curr_edp = []
    bar_optimal_edp = []
    bar_base_edp = []
    base = []
    prediction_map = {}
    print("Overall Accuracy", rfc.score(x_test, y_test))
    for [bench, x_test, y_test, edp_test] in bench_data:        
        print(bench)
        print("Classifier Accuracy\t", rfc.score(x_test, y_test))         
        curr_edp = 0
        optimal_edp = 0
        base_edp = 0
#        print(bench)
#        print("Predicted\t\tCorrect")
        for x, y, edp_line in zip(x_test, y_test, edp_test):
            p = int(rfc.predict(x.reshape(1, -1))[0])
            if p not in prediction_map:
                prediction_map[p] = 1
            else:
                prediction_map[p] = prediction_map[p] + 1
                
            curr_cache = cache_map[int(rfc.predict(x.reshape(1, -1))[0])]
            curr_edp = curr_edp + edp_map[bench + "_" + edp_line[3] + "_" + curr_cache]
            optimal_edp = optimal_edp + float(edp_line[2])
            base_edp = base_edp + float(edp_line[1])
#            if (bench == "gcc" or bench == "namd"):
#                print("%2d %20s\t%2d %20s %5.8f" % (int(rfc.predict(x.reshape(1, -1))[0]), curr_cache, int(y), cache_map[y], edp_map[bench + "_" + edp_line[3] + "_" + curr_cache] - float(edp_line[2])))
          
        bar_bench.append(bench)
        bar_curr_edp.append(curr_edp/base_edp)
        bar_optimal_edp.append(optimal_edp/base_edp)
        bar_base_edp.append(base_edp)
        base.append(1.0)
        print("Classification EDP\t", bar_curr_edp[-1])
        print("Optimal EDP\t\t", bar_optimal_edp[-1])
        
    bar_bench.append("Average")
    bar_curr_edp.append(np.average(bar_curr_edp))
    bar_optimal_edp.append(np.average(bar_optimal_edp))
    base.append(1.0)
    print("Average")
    print("Classification EDP\t", bar_curr_edp[-1])
    print("Optimal EDP\t\t", bar_optimal_edp[-1])
    
    ax = plt.subplot(111)
    w = 0.3
    x = np.arange(len(bar_curr_edp))
    ax.bar(x - w, bar_curr_edp,    width=w, color='b', align='center')
    ax.bar(x    , bar_optimal_edp, width=w, color='g', align='center')
    ax.bar(x + w, base,            width=w, color='r', align='center')

    plt.xticks(x, bar_bench, rotation='vertical')
    plt.legend(['Classification EDP', 'Optimal EDP', 'Base EDP'], loc=9, bbox_to_anchor=(1.2, 1), ncol=1)
    plt.ylabel('EDP')
    plt.title('EDP Comparison')
    plt.savefig("EDP Comparison using SMOTE.png", dpi=1000, bbox_inches='tight')
    plt.show()
    print("Predictions")
    count = 0
    for p in list(prediction_map.keys()):
        print(p, cache_map[p], prediction_map[p])
        count = count + prediction_map[p]
    print("total %d" % count)
    
    

def plotClassifierComparison(X_train, Y_train, X_test, Y_test):
    classifiers = [
#        GradientBoostingRegressor(n_estimators=84, learning_rate=0.1, max_depth=9),
        KNeighborsClassifier(5),
        DecisionTreeClassifier(max_depth=18),
        RandomForestClassifier(max_depth=14, n_estimators=15, max_features=6), 
        BaggingClassifier(n_estimators=84),
        MLPClassifier(hidden_layer_sizes=[150, 150], activation='tanh', solver='adam', learning_rate='invscaling'),
        AdaBoostClassifier(n_estimators=13),
        GaussianNB(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1)]
        
    names = [
#        "GradientBoostingRegressor",
        "Nearest Neighbors", 
        "Decision Tree", 
        "Random Forest", 
        "Bagging",
        "Neural Net", 
        "AdaBoost",
        "Naive Bayes", 
        "Linear SVM", 
        "RBF SVM"]
    score = []
    for clf, name in zip(classifiers, names):
        s = clf.fit(X_train, Y_train).score(X_test, Y_test)
        score.append(s)
    
    
    plt.bar(range(len(score)), score, )
    plt.xticks(range(len(score)), names, rotation='vertical')
    plt.ylabel('Classifier Accuracy')
    plt.title('Classifier Accuracy Comparison')
    plt.savefig("Classifier Accuracy Comparison.png", dpi=1000, bbox_inches='tight')
    plt.show()
    
def plotNumberFeaturesComparison(x, classifiers, names):
    data, labels, feature_names, edp = generate_data(DATASET_PATH+'master.csv')
    print("Accuracy", "Best Number of features")
    for clf, name in zip(classifiers, names):
        print(name)
        score = []
        max_score = 0.0
        features = 0
        for i in x:
            if (name == "Random Forest"):
                clf = RandomForestClassifier(max_depth=14, n_estimators=15, max_features=min(i, 6))

            data1, var_mask, idx_mask = feature_selection_experiment(data, labels, feature_names, k=i)
            (X_train, Y_train), (X_test, Y_test), (edp_train, edp_test) = split_data(data1, labels, edp)
            rfc = clf.fit(X_train, Y_train)
            s = rfc.score(X_test, Y_test)
            if s > max_score:
                max_score = s
                features = i
            score.append(s) 
        print(max_score, features)
        plt.plot(x, score)
        
    plt.legend(names, loc=9, bbox_to_anchor=(1.2, 1), ncol=1)
    plt.title("Changing the number of features used")
    plt.xlabel("Number of features")
    plt.ylabel("Classifier Accuracy")
    plt.savefig("Feature Selection.png", dpi=1000, bbox_inches='tight')    
    plt.show()
    
def thresholdLabels(labels, threshold=10):
    freq = {}    
    for i in labels:
        if i not in freq:
            freq[i] = 1
        else:
            freq[i] = freq[i] + 1
#    print(freq)

    new = 43
    for i in list(freq.keys()):
        if freq[i] < threshold:
#            print("Feature %d" % i)
            for j in range(len(labels)):
                if labels[j] == i:
                    labels[j] = new
                else:
                    labels[j] = labels[j]
                    
    freq = {}    
    for i in labels:
        if i not in freq:
            freq[i] = 1
        else:
            freq[i] = freq[i] + 1
#    print(freq)
        
    return labels
if __name__ == '__main__':
    start_time = time.time()

    data, labels, feature_names, edp = generate_data(DATASET_PATH+'master.csv')
    data, var_mask, idx_mask = feature_selection_experiment(data, labels, feature_names, k=10)
    (X_train, Y_train), (X_test, Y_test), (edp_train, edp_test) = split_data(data, labels, edp)
    
    
    freq = {}
    for i in labels:
        if i not in freq:
            freq[i] = 1
        else:
            freq[i] = freq[i] + 1
    print(freq)
    
    cache_map = {}
    with open(DATASET_PATH+'key.csv') as f:
        reader = csv.reader(f)
        master = list(reader)
    master = np.array(master)
    for i in range(len(master)):
        cache_map[int(master[i, 1])] = master[i, 0]


    print("number of cache occurences")  
    count = 0
    for i in list(freq.keys()):
        count = count + freq[i]
        print(i, cache_map[i], freq[i])
        
    print("total %d" % count)

#    rfc = RandomForestClassifier(max_depth=14, n_estimators=14, max_features=5).fit(X_train, Y_train)
#    score = rfc.score(X_test, Y_test)
#    print(score) 
    
    
#    FindBestClassifier(X_train, Y_train, X_test, Y_test) 
#    NeuralNetworkExperiment(X_train, Y_train, X_test, Y_test)



    plotClassifierComparison(X_train, Y_train, X_test, Y_test)
    
    
    c = RandomForestClassifier(max_depth=14, n_estimators=15, max_features=6)
    plotEDPComparison(X_train, X_test, edp_train, edp_test, var_mask, idx_mask, c)

    classifiers = [
        RandomForestClassifier(max_depth=14, n_estimators=15, max_features=6), 
#        GradientBoostingRegressor(n_estimators=84, learning_rate=0.1, max_depth=9),
#        KNeighborsClassifier(5),
#        DecisionTreeClassifier(max_depth=18),
#        BaggingClassifier(n_estimators=84),
#        MLPClassifier(hidden_layer_sizes=[150, 150], activation='tanh', solver='adam', learning_rate='invscaling'),
#        AdaBoostClassifier(n_estimators=13),
        GaussianNB()
        ]
    names = [
        "Random Forest", 
#        "Gradient Boosting",
#        "Nearest Neighbors", 
#        "Decision Tree", 
#        "Bagging",
#        "Neural Net", 
#        "AdaBoost",
        "Naive Bayes"]
#    for c in classifiers:
#        print(c)
#        plotEDPComparison(X_train, X_test, edp_train, edp_test, var_mask, idx_mask, c)

        
    
    
    plotNumberFeaturesComparison(range(2, 50), classifiers, names)


    data, labels, feature_names, edp = generate_data(DATASET_PATH+'master.csv')
    data1, var_mask, idx_mask = feature_selection_experiment(data, labels, feature_names, k=10)
    (X_train, Y_train), (X_test, Y_test), (edp_train, edp_test) = split_data(data1, labels, edp)
    feature_names = feature_names[var_mask]
    feature_names = feature_names[idx_mask]
    print(feature_names)
    

    for clf, name in zip(classifiers, names):
        rfc = clf.fit(X_train, Y_train)
        s = rfc.score(X_test, Y_test)
        print(name, s)

    
#    14
#    c = RandomForestClassifier(max_depth=8, n_estimators=15, max_features=6).fit(X_train, Y_train)
#    print(c.score(X_test, Y_test))
#    tr = c.estimators_[0]
#    print(tr)
#    
#
#    # Create DOT data
#    from sklearn import datasets
#    iris = datasets.load_iris()
#    X = iris.data
#    y = iris.target
#    dot_data = export_graphviz(tr, out_file=None, filled=True, rounded=True, special_characters=True)
#    
#    # Draw graph
#    graph = pydotplus.graph_from_dot_data(dot_data)  
#    
#    # Show graph
#    Image(graph.create_png())
#    graph.write_png("aaa.png")
    
    
    print("--- %s seconds ---" % (time.time() - start_time))






#    for c in rfc.estimators_:
#        print(c.score(X_test, Y_test))
    
#    feature_names = feature_names[var_mask]
#    feature_names = feature_names[idx_mask]
#    print(feature_names)

        
#        print(d.shape)

#    data, labels = SMOTE().fit_sample(data, labels)
#    (X_train, Y_train), (X_test2, Y_test2) = split_data(data, labels)

#    with open(DATASET_PATH + 'mapping.csv') as f:
#        reader = csv.reader(f)
#        master = list(reader)
#    master = np.array(master)
#    edp_dic = dict(zip(master[:, 0], master[:, 1].astype(float)))
#    
#    for bench in os.listdir(RAWDATA_PATH):
#        simpointDirs = os.listdir(RAWDATA_PATH + bench) 
#        for simpointName in simpointDirs:
#            configDir = os.listdir(RAWDATA_PATH + bench + "/" + simpointName)
#            for config in configDir:
#                path = RAWDATA_PATH + bench + "/" + simpointName + "/" + config
#        
#    rfc = RandomForestClassifier(max_depth=14, n_estimators=14, max_features=5).fit(data, labels)

            
  # Trial for trying to use all other for testing .              
#    for bench1 in os.listdir(BENCHMARK_PATH):
#        with open(BENCHMARK_PATH + bench1) as f:
#            reader = csv.reader(f)
#            master = list(reader)
#        master = np.array(master)
#        master = master[1:, :]
#        X_test = master[:, :-1]
#        Y_test = master[:, -1]
#        X_train = np.zeros((0, X_test.shape[1]))
#        Y_train = []
#        for bench2 in os.listdir(BENCHMARK_PATH):
#            if bench1 == bench2:
#                continue
#            with open(BENCHMARK_PATH + bench1) as f:
#                reader = csv.reader(f)
#                master = list(reader)
#            master = np.array(master)
#            master = master[1:, :]
#            X = master[:, :-1]
#            Y = master[:, -1]
#            X_train = np.vstack((X_train, X))
#            if (len(Y_train) == 0):
#                Y_train = Y 
#            else:
#                Y_train = np.hstack((Y_train, Y))     
#            
#        X_train = np.nan_to_num(X_train.astype(float))
#        X_test = np.nan_to_num(X_test.astype(float))
#        labels = np.vstack((Y_train, Y_test))
#        print(bench1)
#        s = RandomForestClassifier(max_depth=14, n_estimators=14, max_features=5).fit(X_train, Y_train)
#        score = s.score(X_test, Y_test)
#        print(score)
   
#            label = bench1.replace('csv', '') + 'labels'
#            print(bench1.replace('.csv', ''))
#            
#            with open(BENCHMARK_PATH + bench1) as f:
#                reader = csv.reader(f)
#                master = list(reader)
#            master = np.array(master)
#            master = master[1:, :]
#            X = master[:, :-1]
#            Y = master[:, -1]
#            print(master.shape)
#            with open(LABEL_PATH + label) as f:
#                reader = csv.reader(f)
#                master = list(reader)
#            master = [x[0].split(' ') for x in master]
#            master = np.array(master).astype(float)
#            print(master.shape)
#    #        print(master.shape)
        
  
#    frequency = {}
#    for i in Y_test:
#        if i not in frequency:
#            frequency[i] = 1
#        else:
#            frequency[i] = frequency[i] + 1
#    print(frequency)
    

 
#    AdaBoostExperiment(X_train, encoded_train, X_test, encoded_test)
#    SVMExperiment(X_train, Y_train, X_test, Y_test)
    
#    KNNExperiment(X_train, encoded_train, X_test, encoded_test) # k = 307 with knn=3

#    for k in range(1, 10):
#        for j in range(1, 10):
#            for i in range(2, 10):
#                ran = RandomForestClassifier(max_depth=j, n_estimators=i, max_features=k)
#                print(ran.fit(X_train, Y_train).score(X_test, Y_test))
