import os
import csv
import numpy as np
#from progress.bar import Bar

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

DATASET_PATH = '/Users/Sam/Desktop/ECE523/Cognitive-Cache-Tuner/data/'

def generate_data():
    ''' Loads master_dataset from .csv file and returns a data array and a vector of labels'''
    with open(DATASET_PATH+'master.csv') as f:
        reader = csv.reader(f)
        master = list(reader)
    all_data = np.array(master[1:]).astype(float)
    feature_names =  np.array(master[0]).astype(str)
    
    feature_names = feature_names[:-1]
    data          = np.nan_to_num(all_data[:,:-1])
    labels        = all_data[:,-1]
    return data, labels, feature_names

def split_data(data, labels):
    '''Splits a single data array into training and testing by randomly selecting indices'''
    train_ind, test_ind = randomize_data_indeces(data.shape[0], percent_training=0.80)
    training = (data[train_ind,:], labels[train_ind])
    testing  = (data[test_ind, :], labels[test_ind])
    return training, testing

def randomize_data_indeces(size, percent_training=0.80):
    '''Generates a exclusive lists of indeces to paritition training and test data '''
    ind = np.array(range(0, size))
    np.random.seed(0)
    np.random.shuffle(ind)
    train_ind = np.array(ind[:int(size*percent_training)])
    test_ind  = np.array(ind[int(size*percent_training):])
    return train_ind, test_ind

def feature_selection_experiment(data, labels, feature_names):
    fs_percentile = SelectKBest(k=10)
    fs_percentile.fit(data, labels)
    idx_mask = fs_percentile.get_support()

    print feature_names[idx_mask]
    return data[:,idx_mask]

def AdaBoostExperiment(X_train, Y_train, X_test, Y_test):
    ensemble_size = [2, 5, 10, 25, 50, 100]
    results = []

    print '\n==Begin AdaBoost Experiment====='
    for ensemble in ensemble_size:
        AdaBoost = AdaBoostClassifier(n_estimators=ensemble)
        AdaBoost.fit(X_train, Y_train)
        score = AdaBoost.score(X_test, Y_test)
        results.append(score)
        print 'Num Estimators: %d\tAccuracy: %f' %(ensemble, score)

def SVMExperiment(X_train, Y_train, X_test, Y_test):
    kernels = ['linear', 'poly', 'rbf']
    slack   = [0.01, 1.0, 2.0]
    degree  = [1, 2, 3]
    results = []
    
    print '\n==Begin SVM Experiment====='
    #bar = Bar('SVM Classifer', max=len(kernels)*len(slack)*len(degree))
    for kernel in kernels:
        for deg in degree:
            for C in slack:
                SupportVectorMachine = SVC(C=C, kernel=kernel, degree=deg)
                SupportVectorMachine.fit(X_train, Y_train)
                accuracy = SupportVectorMachine.score(X_test, Y_test)
                results.append(accuracy)
                print 'Kernel: %s\tDegree: %d\tC: %f\tAccuracy: %f' \
                        %(kernel, deg, C, score)
                #bar.next()

def NeuralNetworkExperiment(X_train, Y_train, X_test, Y_test):
    num_labels = int(np.max(np.concatenate((Y_train,Y_test),axis=0))+1)
    Y_train = np_utils.to_categorical(Y_train,num_labels)
    Y_test  = np_utils.to_categorical(Y_test, num_labels)

    network_depth = [1,2]
    num_neurons   = [2, 10, 50, 100, 250]
    network_reg   = [False, True]
    results = []

    print '\n==Begin Neural Network Experiment====='
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
                print 'Depth: %d\tNeurons: %d\tDropout: %r\tAccuracy: %f' \
                        %(depth, layer_size, reg, score[1])

def KNNExperiment(X_train, Y_train, X_test, Y_test):
    num_neighbors = [5, 10, 50, 100]
    results = []

    print '\n==Begin KNN Experiment====='
    for neighbors in num_neighbors:
        NearestNeighbor = KNeighborsClassifier(n_neighbors=neighbors)
        NearestNeighbor.fit(X_train, Y_train)
        score = NearestNeighbor.score(X_test, Y_test)
        results.append(score)
        print 'Num Neighbors: %d\tAccuracy: %f' %(neighbors, score)

if __name__ == '__main__':
    data, labels, feature_names = generate_data()
    data = feature_selection_experiment(data, labels, feature_names)

    (X_train, Y_train), (X_test, Y_test) = split_data(data, labels)

    #AdaBoostExperiment(X_train, Y_train, X_test, Y_test)
    #SVMExperiment(X_train, Y_train, X_test, Y_test)
    #NeuralNetworkExperiment(X_train, Y_train, X_test, Y_test)
    KNNExperiment(X_train, Y_train, X_test, Y_test)

