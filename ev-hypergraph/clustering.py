import argparse
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import linalg
from scipy.cluster.vq import vq, kmeans, whiten
from definitions import Attribute, Category

def getH(HG): #incidence matrix
    H = np.zeros((len(HG.nodes), len(HG.edges)))
    nodes = list(HG.nodes)
    for idx, node in enumerate(nodes):
        for edge in node.edges:
            H[idx, edge.id] = 1
    return H

def getW(HG): #edge weight diag matrix
    W = np.zeros((len(HG.edges), len(HG.edges)))
    for edge in HG.edges:
        W[edge.id, edge.id] = edge.weight
    return W
def getDv(HG):
    return np.diag([x.getDegree() for x in HG.nodes])


def run(HG, args, k = 3):
    N = args.nodes
    H = getH(HG)
    W = getW(HG)
    Ht = np.transpose(H)
    Dv = getDv(HG)
    A = np.dot(np.dot(H, W), Ht) - Dv #adjacency matrix

    Dvp = linalg.fractional_matrix_power(Dv, -1/2)# -- Replaced by the following: (raises Dv to the -1/2)


    L =(.5)*(np.identity( Dvp.shape[1]) - np.dot(np.dot(Dvp, A), Dvp))
    eigw, eigv = linalg.eigh(L, eigvals=(1,k))


    #plt.scatter(eigv[:,0],eigv[:,1], c="r", alpha=0.3)

    #K-means algorithm
    data = whiten(eigv)
    centroids, _ = kmeans(data, k)
    idx, _ = vq(data, centroids)



    coalitionArray = []
    completeCoals = []
    for i in range(k):
        lcap = [0]
        lrel = [0]
        ldis = [0]
        coalitionArray.append(set())
        for node in np.array(list(HG.nodes))[idx==i]:
            coalitionArray[i].add(node)
            lcap.append(lcap[-1] + node.EV.attributes[Attribute.capacity])
            lrel.append(np.mean([o.EV.attributes[Attribute.reliability] for o in coalitionArray[i]]))
            ldis.append(ldis[-1] + node.EV.attributes[Attribute.discharge])
            if(lcap[-1] >= args.capacity and ldis[-1] >= args.discharge):
                break
        if(lcap[-1] >= args.capacity and ldis[-1] >= args.discharge):
            completeCoals.append(coalitionArray[i])

    if(not completeCoals):
        return None
    else:
        coalition = min( completeCoals, key=lambda x: len(x))
    #if(lcap[-1] < args.capacity and ldis[-1] < args.discharge):


    #plt.plot(data[:,0], data[:,1], ".")
    if(args.plot):
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        #ax.scatter(data[idx==0,0], data[idx==0,1], data[idx==0,2], c='b')
        #ax.scatter(data[idx==1,0], data[idx==1,1], data[idx==1,2], c='r')
        #ax.scatter(data[idx==2,0], data[idx==2,1], data[idx==2,2], c='g')

        #plt.subplot(2,1,1)
        #plt.plot(data[idx==0,0],data[idx==0,1],'ob',
        #     data[idx==1,0],data[idx==1,1],'or')
        #plt.plot(centroids[:,0],centroids[:,1],'sg', markersize=8)
        #points = np.random.randint(N, size=N/2)
        #for i in range(N):
        #    plt.text(data[i,0], data[i,1], HG.nodes[i].EV.describe())

        fig = plt.figure(4)
        fig.suptitle("Clustering")
        plt.subplot(3,1,1)
        plt.ylabel('Capacity')
        plt.plot(range(len(lcap)), lcap, "b")
        plt.axhline(y=args.capacity, color="r")
        plt.ylim(0,max(args.capacity*1.1, lcap[-1]*1.1))
        plt.subplot(3,1,2)
        plt.ylabel('Reliability')
        plt.plot(range(len(lrel)), lrel, "r")
        plt.subplot(3,1,3)
        plt.ylabel('Discharge')
        plt.plot(range(len(ldis)), ldis, "g")
        plt.ylim(0,max(args.discharge*1.1, ldis[-1]*1.1))
        plt.axhline(y=args.discharge, color="r")
        if(args.writepng):
            plt.savefig("Summary/clustering.png")
        else:
            plt.show()

    return coalition
