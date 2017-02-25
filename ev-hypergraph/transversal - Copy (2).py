import argparse
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
from sim import Hypergraph
from definitions import Attribute, Category

import itertools
import copy
def run(HG, args):
    #HG.edges = sorted(HG.edges, key=lambda x: x.getDegree());
    #S = minimalHS(HG)

    HG_test = copy.deepcopy(HG)
    alg3(HG_test)

    lcap = [0]
    lrel = [0]
    ldis = [0]

    coalition = set()
    oldlen=-1
    k = 0
    while(len(coalition) != oldlen or len(coalition) ==0 ):
        oldlen=len(coalition)
        S = enumerateHS(HG, k, set())
        for frozen in S:
            for node in frozen:
                if(node in coalition):
                    continue
                coalition.add(node)
                lcap.append(lcap[-1] + node.EV.attributes[Attribute.capacity])
                lrel.append(np.mean([o.EV.attributes[Attribute.reliability] for o in coalition]))
                ldis.append(ldis[-1] + node.EV.attributes[Attribute.discharge])
                if(lcap[-1] >= args.capacity and ldis[-1] >= args.discharge):
                    break
            if(lcap[-1] >= args.capacity and ldis[-1] >= args.discharge):
                break
        if((lcap[-1] >= args.capacity and ldis[-1] >= args.discharge) or k > 5):
            break

        k = k +1
    if(args.plot):
        fig = plt.figure(3)
        fig.suptitle("Transversal")
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
            plt.savefig("Summary/transversal.png")
        else:
            plt.show()
    return coalition

    # Add e
    # update the set of generalized nodes
    # express E1 as set of generalized nodes
    # Compute T = Tr(e1)
    # add_next_hyperedge(T, E2)

    # procedure 4:

def enumerateHS(HG, k, X):
    E = list(HG.edges)
    if not E: #E is empty
        return frozenset(X);
    elif k == 0:
        return set()
    else:
        e= E[0]
        S = set()
        for u in e.nodes:
            Vn = HG.nodes - set([u])
            En = HG.edges - set(u.edges)
            HGn = Hypergraph(Vn, En)
            enumHGn = enumerateHS(HGn, k-1, X|set([u]) )
            if type(enumHGn) is frozenset:
                S.add(enumHGn)
            else:
                S = S|enumHGn
        return S

def minimalHS(HG):
    for k in range(len(HG.nodes)+1):
        S = enumerateHS(HG, k, set())
        if(len(S)>0):
            break
    return S


class GeneralizedNode(object):
    """docstring for GeneralizedNode."""
    count = 1
    def __init__(self, nodes):
        if(type(nodes) is set):
            nodes = frozenset(nodes)
        elif(type(nodes) is not frozenset):
            raise TypeError
        self.__nodes = nodes
        self._id = GeneralizedNode.count
        GeneralizedNode.count += 1

    def nodes(self):
        return self.__nodes

    def __hash__(self):
        return self.__nodes.__hash__()
    def __repr__(self):
        return "g" + str(self._id) #+ ": " + str(set(self.__nodes)) #+ "\n"


def alg3(HG):

    HG.nodes = [1,2,3,4,5]
    HG.edges = [set([1,2,3]),
                set([3,4,5]),
                set([1,5]),
                set([2,5])]

    print("======Hypergraph=======")
    for e in HG.edges:
        print("Edge: ", e)
    print("=========END===========")

    generalized_nodes = set()
    edge_iter = iter(HG.edges)

    #ADD E0
    e0 = next(edge_iter)

    #g = GeneralizedNode(frozenset([v for v in HG.nodes
    #                                 if e0 in (set(v.edges) & HG.edges)]))

    g = GeneralizedNode(frozenset(e0))

    #UPDATE SET OF GENERALIZED NODES (INITIALIZE IT)
    generalized_nodes.add(g)

    # Express E0 as set of generalized_nodes = list(generalized_nodes)[0]

    #Compute T = Tr(e0):

    T = set(generalized_nodes)


    e1 = next(edge_iter)
    add_next_hyperedge(edge_iter, T, e1, generalized_nodes)


def add_next_hyperedge(edge_iter, T, E, generalized_nodes):
    #print("Adding edge: ", E)
    print()
    #print("Old generalized: ", str(generalized_nodes))
    generalized_nodes, E_as_generalized, T_as_generalized, split = update_generalized_nodes(E, T, generalized_nodes)

    print("T: " + str(T))
    print("New edge: " + str(E))
    print("Generalized: ")
    for g in generalized_nodes:
        print("\t", g, ": ", g.nodes())
    print("E generalized: ", E_as_generalized)
    print("T generalized: ", T_as_generalized)

    l = 0
    if split:
        offsprings = generate_next_transversal(T_as_generalized, E_as_generalized)
    else:
        offsprings = [T]
    print("TxE: ", offsprings)
    print()

    e_next = next(edge_iter, None)

    for Tn in offsprings:
        if Tn is None:
            raise ValueError


        if e_next is None:
            print("RESULTS: " + str(Tn))

        else:
            edge_iter, new_edge_iter = itertools.tee(edge_iter, 2)
            add_next_hyperedge(new_edge_iter, Tn, e_next, generalized_nodes)
            l = l+1



def CartesianProduct(A, B):
    out = []

    for a in A:
        for b in B:
            out.append(a | b)
    return out

def generate_next_transversal(T, E):
    E = [set([e]) for e in E]

    product = list()
    """if(len(T) == 1):
        T = [set(next(iter(T)))]
    else:
        #T = itertools.product(*T)"""
    print(T)
    T = [set(t) for t in T]

    print("T: \t", T)
    print("E: \t", E)

    #product = [a | E for a in (t for t in T)]
    '''
    for t in T:
        p = itertools.product(t,E)
        p = [set(i) for i in p]
        product.extend(p)
    '''
    #product = itertools.product(T,E)

    product = CartesianProduct(T,E)

    #product = list(product)

    #product = [set(p) for p in product]
    print("P: \t", product)



    #for a in product:
    #    print(a)
    out = []

    for a in product:
        for b in product:
            if a == b:
                continue
            elif a > b:
                break
        else:
            if a not in out:
                out.append(a)
    return out

def update_generalized_nodes(E, T, generalized_nodes):
    split = False
    E_as_generalized = set()
    T_as_generalized = set()

    new_nodes = set(E)
    #update set of generalized nodes:
    new_generalized_nodes = set()

    X_new = new_nodes.copy() #here we will store nodes we haven't discovered


    for Xf in generalized_nodes:

        X = Xf.nodes()
        X_new -= X

        newTr = []

        if(X & new_nodes == set()):
            X1 = Xf#GeneralizedNode(X)
            new_generalized_nodes.add(X1)
            for t in T:
                if (Xf == t):
                    T_as_generalized.add(frozenset([X1]))
        elif X < new_nodes:
            X1 = Xf#GeneralizedNode(X)
            new_generalized_nodes.add(X1)
            E_as_generalized.add(X1)
            for t in T:
                if (Xf == t):
                    T_as_generalized.add(frozenset([X1]))
        else:
            split = True
            X2 = GeneralizedNode(X & new_nodes)
            X1 = GeneralizedNode(X - X2.nodes())

            E_as_generalized.add(X2)

            new_generalized_nodes.add(X1)
            new_generalized_nodes.add(X2)


            for t in T:
                if (Xf == t):
                    if len(T_as_generalized) == 0:
                        T_as_generalized.add(frozenset([X1]))
                        T_as_generalized.add(frozenset([X2]))
                    else:
                        myset = CartesianProduct(set([frozenset([X1]), frozenset([X2])]), T_as_generalized)
                        T_as_generalized = set(myset)
    if(X_new):
        X_new = GeneralizedNode(X_new)
        new_generalized_nodes.add(X_new)
        E_as_generalized.add(X_new)
    #T_as_generalized.add(GeneralizedNode(X_new))
    #T_as_generalized = set(T)
    return new_generalized_nodes, E_as_generalized, T_as_generalized, split
