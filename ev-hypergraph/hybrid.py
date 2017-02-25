import argparse
import random
import greedy
import clustering
import transversal
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg

from definitions import Attribute, Category
import copy


def run(HG, args):


	transversalperc = 0.3
	transargs = copy.deepcopy(args)
	transargs.capacity = transversalperc * args.capacity
	transargs.discharge = transversalperc * args.discharge
	transargs.reliability = transversalperc * args.reliability
	transargs.plot = False
	coalition = transversal.run(HG.prune(lambda e: e.category==Category.exhigh, "this"), transargs)

	sl = sorted(HG.nodes, key=lambda x: x.getDegree(), reverse=True);
	lcap = [0]
	lrel = [0]
	ldis = [0]

	for node in sl:
		coalition.add(node)
		lcap.append(lcap[-1] + node.EV.attributes[Attribute.capacity])
		lrel.append(np.mean([o.EV.attributes[Attribute.reliability] for o in coalition]))
		ldis.append(ldis[-1] + node.EV.attributes[Attribute.discharge])

		if(lcap[-1] >= args.capacity and ldis[-1] >= args.discharge):
			break

	if(args.plot):
		fig = plt.figure(1)
		fig.suptitle("Hybrid")
		plt.subplot(3,1,1)
		plt.ylabel('Capacity')
		plt.plot(range(len(lcap)), lcap, "b")
		plt.ylim(0,max(args.capacity*1.1, lcap[-1]*1.1))
		plt.axhline(y=args.capacity, color="r")
		plt.subplot(3,1,2)
		plt.ylabel('Reliability')
		plt.plot(range(len(lrel)), lrel, "r")
		plt.subplot(3,1,3)
		plt.ylabel('Discharge')
		plt.plot(range(len(ldis)), ldis, "g")
		plt.ylim(0,max(args.discharge*1.1, ldis[-1]*1.1))
		plt.axhline(y=args.discharge, color="r")
		if(args.writepng):
			plt.savefig("Summary/hybrid.png")
		else:
			plt.show()

	return coalition
