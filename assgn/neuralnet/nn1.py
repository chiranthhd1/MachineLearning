from numpy import *
from sklearn.neural_network import MLPClassifier

data = [] # training data set
labels = [] # traning label set
test = [] # test data set

with open ('TestData.csv') as test:
    row = 0 # skip the first line
    for line in test:
        testSamples = line.strip().split(",") 
        if(row >0):
            test.append([float(num) for num in testSamples])
        row +=1

with open ('TrainData.csv') as train:
    row = 0 # skip the first line
    for line in train:
        trainSamples = line.strip().split(",") 
        if(row > 0):
            data.append([float(num) for num in trainSamples[:-1]])
            labels.append(trainSamples[-1])
        row +=1



clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(data,labels) 
MLPClassifier(activation='relu', algorithm='l-bfgs', alpha=1e-05,
       batch_size=200, beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
y_pred = clf.predict(test)


print (y_pred)