import pybrain
import time
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from numpy import *
import xlrd
traindata=[]
testdata=[]
label=[]
downsize=150
def pca(data,dem):
 meanValue=mean(data,axis=0)
 meanRemoved=data-meanValue
 covMat=cov(meanRemoved,rowvar=0)
 eigVals,eigVects=linalg.eig(mat(covMat))
 eigValInd=argsort(eigVals)
 eigValInd=eigValInd[:-(dem+1):-1]
 redEigVects=eigVects[:,eigValInd]
 lowDData=meanRemoved*redEigVects
 #reconMat=(lowDData*redEigVects.T)+meanVals
 return lowDData
testfile=xlrd.open_workbook("TestData.xls")
testsheet=testfile.sheet_by_name("Sheet1")
for i in range(0,300):
 tmp=[]
 for j in range(0,150):
  tmp.append(testsheet.cell(i,j).value)
 testdata.append(tmp)
trainfile=xlrd.open_workbook("TrainData.xls")
trainsheet=trainfile.sheet_by_name("Sheet1")
for i in range(0,2400):
 tmp=[]
 for j in range(0,150):
  tmp.append(trainsheet.cell(i,j).value)
 traindata.append(tmp)
#traindata=pca(traindata,downsize)
#print traindata
trainlabel=xlrd.open_workbook("TrainLabel.xls")
labelsheet=trainlabel.sheet_by_name("Sheet1")
for i in range(0,2400):
 label.append(labelsheet.cell(i,0).value)
net=buildNetwork(downsize,50,1)
ds=SupervisedDataSet(downsize,1)
#print len(ds)
#print len(traindata)
#print len(label)
for i in range(0,2400):
 ds.addSample(traindata[i],label[i])
#trainer=BackpropTrainer(net,ds,verbose=True)
print "loaded"
trainer=BackpropTrainer(net,ds)
#trainer.train()
start=time.clock()
trainer.trainUntilConvergence(maxEpochs=50)
totaltime=(time.clock()-start)
print("Spent time:"+str(totaltime))
print "result:"
out=SupervisedDataSet(downsize,1)
for i in range(0,300):
 out.addSample(testdata[i],0)
net.reset()
out=net.activateOnDataset(out)
print out
