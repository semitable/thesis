#!/usr/bin/env python
import argparse
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.cluster.vq import vq, kmeans, whiten


import transversal
import clustering
import greedy
import hybrid
import sample
import time
import timeit

from definitions import Attribute, Category, assignCategory, gaussianAttributes, booleanAttributes, getWeight

class EVs:

    def __init__(self):
        self.attributes = {}
        self.genAttributes()
        self.characteristics = {}
        for c in Category:
            self.characteristics[c] = []
        self.genCharacteristics()

    def genTimeSlots(self):
        self.slots = np.random.randint(2, size=24)
    def genAttributes(self):
        for (name, (m , sigma)) in list(gaussianAttributes.items()):
            self.attributes[name] = random.gauss(m, sigma)
        for (name, probability) in list(booleanAttributes.items()):
            self.attributes[name] = random.random() < probability

    def genCharacteristics(self):
        for (name, value) in list(self.attributes.items()):
            self.characteristics[assignCategory(name, value)].append(name)


    def getThreshold(self, name):
        (m, sigma) = self.gaussianAttributes.get(name)
        return (m-1*sigma, m+1*sigma)
    def describe(self):
        return "H:" + str(self.characteristics[Category.high]) + "\nM:" + str(self.characteristics[Category.medium])+ "\nL:" + str(self.characteristics[Category.low])



class Node:
    count = 0
    def __init__(self, EV):
        self.EV = EV
        self.edges = []
        self.id = Node.count
        Node.count = Node.count + 1
    def __repr__(self):
        return "u:" + str(self.id)

    def addEdge(self, edge):
        self.edges.append(edge)
    def getDegree(self):
        return sum([x.weight for x in self.edges])

class Edge:
    count = 0
    def __init__(self, name, weight, category):
        self.nodes = []
        self.name = name
        self.category = category
        self.id = Edge.count
        self.weight =weight
        Edge.count = Edge.count + 1
    def __repr__(self):
        return str(self.name) + " in " + str(self.category)
    def addNode(self, node):
        self.nodes.append(node)
    def remNode(self, node):
        self.nodes.remove(node)
    def getDegree(self):
        return len(self.nodes)

class Hypergraph:
    def __init__(self, V = set(), E = set()):
        self.nodes = V
        self.edges = E

    def addNode(self, node):
        self.nodes.add(node)

    def addToEdge(self, node, edge):
        edge.addNode(node)
        node.addEdge(edge)

    def getEdge(self, name, weight = 0, category=Category.empty): #Returns an Edge object for the "name". Creates one if it doesn't exist
        #if(name not in self.edges):
        #    self.edges[name] = Edge(name, weight, category)
        #return self.edges[name]
        e = next((x for x in self.edges if x.name == name and x.category==category), None)
        if e == None:
            e = Edge(name, weight, category)
            self.edges.add(e)
        return e

    def prune(self, f, keep="all"):
        nodesInEdges = set()
        newEdges = set(filter(f, list(self.edges)))
        for e in newEdges:
            nodesInEdges = nodesInEdges | set(e.nodes)
        newNodes = nodesInEdges & set(self.nodes)
        if(keep == "all"):
            newHG = Hypergraph(V = newNodes, E = self.edges)
        elif(keep == "other"):
            newHG = Hypergraph(V = newNodes, E = self.edges - newEdges)
        else:
            newHG = Hypergraph(V = newNodes, E = newEdges)

        #print("New size: " + str(len(newHG.nodes)), flush=True)
        return newHG



def rateCoalition(args, coalition, time, title="---"):
    if(args.fulltest):
        return
    print("=========================")
    print("----" + title + "----")
    print("=========================")
    print("Coalition:")
    print(coalition)
    print("Coalition size: " + str(len(coalition)))
    print("Capacity requirement achieved: " + str(sum([x.EV.attributes[Attribute.capacity] for x in coalition]) >= args.capacity))
    print("Discharge requirement achieved: " + str(sum([x.EV.attributes[Attribute.discharge] for x in coalition]) >= args.discharge))
    print("Reliability: " + str(sum([x.EV.attributes[Attribute.reliability] for x in coalition])/len(coalition)))
    print("Time Elapsed: " + str(int(time * 1000))+ "ms", flush = True)

def Generate(N):
    R = set()
    Edge.count = 0
    Node.count = 0
    HG = Hypergraph(set(), set())
    EVlist = []
    for i in range(N):
        EVlist.append(EVs())
    for ev in EVlist:
        node = Node(ev)
        HG.addNode(node)
        for acategory, characteristic in ev.characteristics.items():
            weight = getWeight(acategory)
            for c in characteristic:
                HG.addToEdge(node, HG.getEdge(c, weight, category=acategory))
    #print(len(HG.nodes))
    return HG


def main(args):
    N = args.nodes
    HG = Generate(N)

    HG = HG.prune(lambda e: e.category==Category.boolean, "all")

    if(args.transversal):
        startTime = time.time()
        R = transversal.run(HG.prune(lambda e: e.category==Category.vhigh, "this"), args)
        elapsed = time.time() - startTime
        rateCoalition(args, R, elapsed, "Transversal")

    #print(HG.edges)
    if(args.clustering):
        startTime = time.time()
        R = clustering.run(HG.prune(lambda e: e.category==Category.vhigh, "all"), args)
        elapsed = time.time() - startTime
        rateCoalition(args, R, elapsed, "Clustering")

    if(args.greedy):
        startTime = time.time()
        R = greedy.run(HG, args)
        elapsed = time.time() - startTime
        rateCoalition(args, R, elapsed, "Greedy")
    if(args.sample):
        startTime = time.time()
        R = sample.run(HG, args)
        elapsed = time.time() - startTime
        rateCoalition(args, R, elapsed, "Sample")

    if(args.hybrid):
        startTime = time.time()
        R = hybrid.run(HG, args)
        elapsed = time.time() - startTime
        rateCoalition(args, R, elapsed, "Hybrid")


    #print (HG.edges)
    #for category in HG.edges:
    #    print(category + ": " + str(HG.edges[category].keys()))

def testchangingk(args):
    repeattimes = 10
    cluster_size = []
    for k in range (2, 15):
        medsize = []
        for j in range (50):
            HG = Generate(args.nodes)
            HG = HG.prune(lambda e: e.category==Category.boolean, "all")
            answer = clustering.run(HG.prune(lambda e: e.category==Category.vhigh, "all"), args, k = k)
            if(answer):
                medsize.append(len(answer))
        cluster_size.append(np.mean(medsize))
    plt.figure()
    fig = plt.figure(5)
    fig.suptitle("Cluster K-Scaling")
    plt.plot(range(2,15), cluster_size)
    plt.show()
def testpruning(args):
    len_trans=0
    len_cluster = 0
    len_greedy = 0
    for k in range(10):
        HG = Generate(args.nodes)
        HG = HG.prune(lambda e: e.category==Category.boolean, "all")
        len_trans = len_trans + len(HG.prune(lambda e: e.category==Category.vhigh, "this").edges)
        len_cluster = len_cluster + len(HG.prune(lambda e: e.category==Category.vhigh or e.category == Category.exhigh, "this").edges)
        len_greedy = len_greedy + len(HG.edges)
    print(len_trans/10)
    print(len_cluster/10)
    print(len_greedy/10)


def scaleonlygreedy(args):
    repeattimes = 4
    Timed = {"Nodes": [],  "Greedy" : [], "Generation":[]}
    for i in range(50000, 1000001, 50000):
        args.nodes = i
        print("Testing with: " + str(i) + " nodes.", flush=True)
        gen_time = 0
        trans_time = 0
        greedy_time = 0
        cluster_time = 0
        for r in range(repeattimes):
            #print("Timing Generation...", flush=True, end="")
            #print("%.3f ms" % Timed.get("Generation")[-1], flush=True)
            gen_time = gen_time + (min(timeit.Timer(lambda: Generate(i)).repeat(repeat=1, number = 1)))
            HG = Generate(i)
            HG = HG.prune(lambda e: e.category==Category.boolean, "all")

            #print("Timing Greedy...", flush=True, end="")
            greedy_time = greedy_time + (min(timeit.Timer(lambda: greedy.run(HG, args)).repeat(repeat=1, number = 2))/2)
            #print("%.2f ms" % Timed.get("Greedy")[-1], flush=True)

        Timed.get("Nodes").append(i)
        Timed.get("Generation").append(gen_time/repeattimes)
        Timed.get("Greedy").append(greedy_time/repeattimes)
        print(Timed)
    plt.figure()
    fig = plt.figure(1)
    fig.suptitle("Time Scaling")
    plt.plot(Timed.get("Nodes"), Timed.get("Greedy"))
    fig = plt.figure(2)
    fig.suptitle("Time Scaling")
    plt.plot(Timed.get("Nodes"), Timed.get("Generation"))

    plt.show()


def fulltest(args):
    N = 80000
    repeattimes = 1
    NoOfHGs = 100
    # Timed = {"Nodes": [], "Generation":[], "Greedy" : [], "Transversal": [], "Clustering" : [], "Hybrid" : []}
    # for i in range(10000, N+1, 5000):
    #     args.nodes = i
    #     print("Testing with: " + str(i) + " nodes.", flush=True)
    #     gen_time = 0
    #     trans_time = 0
    #     greedy_time = 0
    #     cluster_time = 0
    #     hybrid_time = 0
    #     for r in range(repeattimes):
    #         #print("Timing Generation...", flush=True, end="")
    #         #gen_time = gen_time + (min(timeit.Timer(lambda: Generate(i)).repeat(repeat=1, number = 1)))
    #         #print("%.3f ms" % Timed.get("Generation")[-1], flush=True)
    #         for hg in range(NoOfHGs):
    #             HG = Generate(i)
    #             HG = HG.prune(lambda e: e.category==Category.boolean, "all")
    #             # print("Timing Transversal...", flush=True, end="")
    #             trans_time = trans_time + (min(timeit.Timer(lambda: transversal.run(HG.prune(lambda e: e.category==Category.exhigh, "this"), args)).repeat(repeat=1, number = 1)))
    #             #print("%.2f ms" % Timed.get("Transversal")[-1], flush=True)
    #
    #             # print("Timing Greedy...", flush=True, end="")
    #             greedy_time = greedy_time + (min(timeit.Timer(lambda: greedy.run(HG, args)).repeat(repeat=1, number = 2))/2)
    #             #print("%.2f ms" % Timed.get("Greedy")[-1], flush=True)
    #
    #
    #             # print("Timing Clustering...", flush=True, end="")
    #             cluster_time = cluster_time + (min(timeit.Timer(lambda: clustering.run(HG.prune(lambda e: e.category==Category.vhigh, "all"), args)).repeat(repeat=1, number = 1)))
    #
    #             hybrid_time = hybrid_time + (min(timeit.Timer(lambda: greedy.run(HG, args)).repeat(repeat=1, number = 1)))
    #             #print("%.2f ms" % Timed.get("Clustering")[-1], flush=True)
    #             #print(Timed.get("Clustering"), flush=True)
    #     Timed.get("Nodes").append(i)
    #     #Timed.get("Generation").append(gen_time/repeattimes)
    #     Timed.get("Greedy").append(greedy_time/(NoOfHGs*repeattimes))
    #     Timed.get("Transversal").append(trans_time/(NoOfHGs*repeattimes))
    #     Timed.get("Clustering").append(cluster_time/(NoOfHGs*repeattimes))
    #     Timed.get("Hybrid").append(hybrid_time/(NoOfHGs*repeattimes))
    #     print(Timed)
    #
    # fig = plt.figure(5)
    # fig.suptitle("Node Scaling")
    # #plt.subplot(3,1,1)
    # #plt.plot(Timed.get("Nodes"), Timed.get("Generation"))
    # plt.subplot(2,2,1)
    # plt.plot(Timed.get("Nodes"), Timed.get("Greedy"), "r")
    # plt.subplot(2,2,2)
    # plt.plot(Timed.get("Nodes"), Timed.get("Transversal"), "g")
    # plt.subplot(2,2,3)
    # plt.plot(Timed.get("Nodes"), Timed.get("Clustering"), "b")
    # plt.subplot(2,2,4)
    # plt.plot(Timed.get("Nodes"), Timed.get("Hybrid"), "b")
    #
    # plt.savefig("Summary/node_scaling.png")

    N = 20000
    args.nodes = N
    Timed = {"Nodes": [], "Generation":[], "Greedy" : [], "Transversal": [], "Clustering" : [], "Hybrid" : []}
    HG = Generate(N)
    HG = HG.prune(lambda e: e.category==Category.boolean, "all")
    for i in range(10000, 300000, 10000):
        args.capacity = i
        print("Testing with: " + str(i) + " capacity goal.", flush=True)
        gen_time = 0
        trans_time = 0
        greedy_time = 0
        cluster_time = 0
        hybrid_time = 0
        for r in range(repeattimes):
            for hg in range(NoOfHGs):
                HG = Generate(N)
                HG = HG.prune(lambda e: e.category==Category.boolean, "all")
                trans_time = trans_time + (min(timeit.Timer(lambda: transversal.run(HG.prune(lambda e: e.category==Category.exhigh, "this"), args)).repeat(repeat=1, number = 1)))
                greedy_time = greedy_time + (min(timeit.Timer(lambda: greedy.run(HG, args)).repeat(repeat=1, number = 2))/2)
                cluster_time = cluster_time + (min(timeit.Timer(lambda: clustering.run(HG.prune(lambda e: e.category==Category.exhigh, "all"), args)).repeat(repeat=1, number = 1)))
                hybrid_time = hybrid_time + (min(timeit.Timer(lambda: greedy.run(HG, args)).repeat(repeat=1, number = 1)))

        Timed.get("Nodes").append(i)
        Timed.get("Greedy").append(greedy_time/(NoOfHGs*repeattimes))
        Timed.get("Transversal").append(trans_time/(NoOfHGs*repeattimes))
        Timed.get("Clustering").append(cluster_time/(NoOfHGs*repeattimes))
        Timed.get("Hybrid").append(hybrid_time/(NoOfHGs*repeattimes))
        print(Timed)
    #print(Timed.get("Nodes"))
    #print(Timed.get("Transversal"))
    fig = plt.figure(6)
    fig.suptitle("Goal Scaling")
    plt.subplot(2,2,1)
    plt.plot(Timed.get("Nodes"), Timed.get("Greedy"), "r")
    plt.subplot(2,2,2)
    plt.plot(Timed.get("Nodes"), Timed.get("Transversal"), "g")
    plt.subplot(2,2,3)
    plt.plot(Timed.get("Nodes"), Timed.get("Clustering"), "b")
    plt.subplot(2,2,4)
    plt.plot(Timed.get("Nodes"), Timed.get("Hybrid"), "b")
    plt.savefig("Summary/goal_scaling.png")
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", help="Number of vehicles that should be generated", type=int, default=10000)
    parser.add_argument("--capacity", help="The goal capacity", type=float, default=10000)
    parser.add_argument("--reliability", help="The reliability goal", type=float, default=0.8)
    parser.add_argument("--discharge", help="The discharge rate goal", type=float, default=1000)

    parser.add_argument("--clustering", help="Run the clustering algorithm's tests", action="store_true")
    parser.add_argument("--greedy", help="Run the greedy algorithm's tests", action="store_true")
    parser.add_argument("--transversal", help="Run the transversal algorithm's tests", action="store_true")
    parser.add_argument("--hybrid", help="Run the hybrid algorithm's tests", action="store_true")

    parser.add_argument("--sample", help="Randomly sample elements", action="store_true")
    parser.add_argument("--plot", help="Plot stuff", action="store_true")
    parser.add_argument("--writepng", help="Plot stuff", action="store_true")
    parser.add_argument("--fulltest", help="Gradually increase node size until given size. Time the results", action="store_true")
    args = parser.parse_args()
    if(args.fulltest):
        fulltest(args)
        #testchangingk(args)
        #scaleonlygreedy(args)
    else:
        #testpruning(args)
        main(args)
