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

    #HG_test = copy.deepcopy(HG)
    #alg3(HG_test)

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


class DFS(object):
    """docstring for DFS."""
    def __init__(self, HG):
        crit = []
        uncov = set()
        CAND = set()

        MinimalTransversals = set()

    def run(S):
        if(not uncov):
            MinimalTransversal.add(frozenset(S))
            return
