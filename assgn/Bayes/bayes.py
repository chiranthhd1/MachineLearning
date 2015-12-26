data = [] # training data set
labels = [] # traning label set
test = [] # test data set

with open ('TestData.csv') as testf:
    row = 0 # skip the first line
    for line in testf:
        testSamples = line.strip().split(",") 
        if(row >0):
            test.append([float(num) for num in testSamples])
        row +=1

with open ('TrainData.csv') as f:
    row = 0 # skip the first line
    for line in f:
        trainSamples = line.strip().split(",") 
        if(row > 0):
            data.append([float(num) for num in trainSamples[:-1]])
            labels.append(trainSamples[-1])
        row +=1



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data,labels).predict(test)

print ( y_pred)

