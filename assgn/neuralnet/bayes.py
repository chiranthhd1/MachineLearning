#from sklearn import datasets
#iris = datasets.load_iris()



data = [] # training data set
labels = [] # traning label set
test = [] # test data set

with open ('TestData.csv') as testf:
    row = 0 # skip the first line
    for line in testf:
        testSamples = line.strip().split(",") # trim whitespaces
        if(row >0):
            test.append([float(num) for num in testSamples]) # convert string to float
        row +=1

with open ('TrainData.csv') as f:
    row = 0 # skip the first line
    for line in f:
        trainSamples = line.strip().split(",") # trim whitespaces
        if(row > 0):
            data.append([float(num) for num in trainSamples[:-1]])
            labels.append(trainSamples[-1])
        row +=1



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data,labels).predict(test)

print ( y_pred)
#print("Number of mislabeled points out of a total %d points : %d"
#	% (iris.data.shape[0],(iris.target != y_pred).sum()))
#Number of mislabeled points out of a total 150 points : 6
