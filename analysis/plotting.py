import argparse
import sys
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import itertools



def plotQs(stateComps, stateVals, actions, qVals):
    """

    qVals should have an axis for each state component and for action
    For example, if there are 3 state components with 4, 2, and 3 different
    values, and 6 possible actions, the shape of qVals would be (4, 2, 3, 6)

    :param list qVals: A list where each element is a snapshot of the Q-table
    """

    combsRemaining = np.prod(len(valSet) for valSet in stateVals)

    labels = [
        saPair for saPair in itertools.product(*(stateVals), actions)
    ]
    def formatLabel(label):
        return "(" + ",".join(label[:-1]) + ") -> " + label[-1]
    labels = [label for label in map(formatLabel, labels)]


    qVals = np.array(qVals)
    qVals = qVals.reshape( (qVals.shape[0], np.prod(qVals.shape[1:])) )
    qVals = qVals.transpose()

    skipIndex = actions.index("nothing")
    for (i, (SA, label)) in enumerate(zip(qVals, labels)):
        if i % len(actions) != skipIndex:
            plt.plot(SA, label=label)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    path = "q.json"
    with open(path, "r") as r:
        history = json.load(r)
    plotQs(history["stateComps"], history["stateVals"],
        history["actions"],history["qVals"])
