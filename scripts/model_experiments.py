import os
import csv
import numpy as np
#from progress.bar import Bar

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

DATASET_PATH = '/Users/Sam/Desktop/ECE523/Cognitive-Cache-Tuner/data/'

def generate_data():
    # load data from master database
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
    fs_percentile = SelectKBest()
    fs_percentile.fit(data, labels)
    idx_mask = fs_percentile.get_support()

    print feature_names[idx_mask]
    return data[:,idx_mask]

def AdaBoostExperiment():
    pass

def SVMExperiment(X_train, Y_train, X_test, Y_test):
    kernels = ['linear', 'poly', 'rbf']
    slack   = [0.01, 1.0, 2.0]
    degree  = [1, 2, 3]
    results = []
    
    #bar = Bar('SVM Classifer', max=len(kernels)*len(slack)*len(degree))
    for kernel in kernels:
        for deg in degree:
            for C in slack:
                SupportVectorMachine = SVC(C=C, kernel=kernel, degree=deg)
                SupportVectorMachine.fit(X_train, Y_train)
                accuracy = SupportVectorMachine.Score(X_test, Y_test)
                results.append(accuracy)
                #bar.next()
    print results

def NeuralNetworkExperiment(X_train, Y_train, X_test, Y_test):
    num_labels = int(np.max(np.concatenate((Y_train,Y_test),axis=0))+1)
    Y_train = np_utils.to_categorical(Y_train,num_labels)
    Y_test  = np_utils.to_categorical(Y_test, num_labels)

    network_depth = [1,2]
    num_neurons   = [2, 10, 50, 100, 250]
    network_reg   = [False, True]
    results = []
    for depth in network_depth:
        for layer_size in num_neurons:
            for reg in network_reg:
                model = Sequential()
                model.add(Dense(layer_size, input_dim=data.shape[1]))
                model.add(Dense(num_labels, activation='softmax'))
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
                model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
                score = model.evaluate(X_test, Y_test, verbose=0)
                results.append(score[1])
    print results

def KNNExperiment():
    pass

if __name__ == '__main__':
    data, labels, feature_names = generate_data()
    data = feature_selection_experiment(data, labels, feature_names)

    (X_train, Y_train), (X_test, Y_test) = split_data(data, labels)

    #AdaBoostExperiment()
    SVMExperiment(X_train, Y_train, X_test, Y_test)
    #NeuralNetworkExperiment(X_train, Y_train, X_test, Y_test)
    #KNNExperiment()

    #for i in range(3000):
    #    print (preds[i], labels[i])
    #print 'Accuracy: %f' %(np.sum(preds==labels[:3000])/float(3000))
