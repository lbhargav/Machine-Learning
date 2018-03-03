import sys
import pandas as pd
import random
import math
import numpy as np

# loading dataset
dataset = pd.read_csv(sys.argv[2], sep = "\t")

#creating dictinary with index as key and x,y as corresponding values
dictId = {}
for i in range (0,len(dataset)):
    id=dataset.iloc[i, 0]
    x = dataset.ix[i, 1]
    y = dataset.ix[i, 2]
    dictId.update({id:[x,y]})
dictCentroids={}

#giving number of clusters as user input
k=int(sys.argv[1])

#writing output as a text file
output=sys.argv[3]
taken=[]
#initializing random centroids for first iteration
i=0
while i<k:
    row=random.randint(1,99)
    if row in taken:
        i=i-1
        continue
    else:

        taken.append(row)
    x = dataset.ix[row, 1]
    y = dataset.ix[row, 2]
    value = [x, y]
    if value not in dictCentroids.values():
        dictCentroids.update({i + 1: [x, y]})
        i = i + 1

    else:
        i=i-1

#finding euclidean distance and allotting corresponding cluster
for i in range(25):
    dictClusters = {}
    dictClusDist={}
    for i in dictCentroids:
        dictClusters.update({i : []})
    dictClusDist = {}
    for i in dictId:
        minDis = sys.maxsize
        for j in dictCentroids:
            euclDis = math.sqrt(pow(float(dictId[i][0])-float(dictCentroids[j][0]),2)+pow(float(dictId[i][1])-float(dictCentroids[j][1]),2))
            if euclDis<minDis:
                minDis=euclDis
                assignedCent=j
        dictClusters[assignedCent].append(i)
        if assignedCent not in dictClusDist:
            dictClusDist.update({assignedCent:[]})
            dictClusDist[assignedCent].append(minDis)
        else:
            dictClusDist[assignedCent].append(minDis)
    dictCentroidsCopy=dictCentroids
    dictCentroids={}
    notChangedFlag=0
    count=0
    for clusterId in dictCentroidsCopy:
     dictCentroids.update({clusterId: dictCentroidsCopy[clusterId]})
    for clusterId in dictCentroidsCopy:
        newX=0
        newY=0
        oldX=dictCentroids[clusterId][0]
        oldY=dictCentroids[clusterId][1]
        if len(dictClusters[clusterId])==0:
            newX=oldX
            newY=oldY
            continue
        for pointId in dictClusters[clusterId]:
            newX=newX+dictId[pointId][0]
            newY = newY + dictId[pointId][1]
        newX=newX/len(dictClusters[clusterId])
        newY = newY / len(dictClusters[clusterId])
        if [newX,newY] in dictCentroidsCopy.values():
            count=count+1
        dictCentroids.update({clusterId:[newX,newY]})
        if count==len(dictClusters):
            notChangedFlag=1
            break
    if notChangedFlag==1:
        break


# Function to calculate SSE
def calSse():
    totalSum=0
    for clusterId in dictClusDist:
        for dist in dictClusDist[clusterId]:
            totalSum=totalSum+(dist)*(dist)

    f = open(sys.argv[3], "w")
    for key, value in dictClusters.items():
        f.write("Cluster "+str(key)+" : "+str(value))
        f.write("\n")
        print(str(key) + " : " + str(value))
    f.write("SSE : "+str(totalSum))
    print("SSE : ")
    print(totalSum)
    f.close()


#Function call to calculate SSE
calSse()



