
import numpy as np

stateComps = ["orientation", "brightness"]
stateVals = [["portrait", "landscape"], ["full", "medium", "low"]]
actionNames = ["nothing", "rotate", "brighten"]

class Bandit(object):
    """
    An agent for contextual bandit problems
    """

    def __init__(self, alpha = 0.1, epsilon = 0.2):
        self._alpha = alpha
        self._epsilon = epsilon

        self._numActions = len(actionNames)
        self._numStates = np.prod([ len(valSet) for valSet in stateVals ])

        self._q = np.zeros( (self._numStates, self._numActions) )

        self._id = -1
        self._actRecord = {}
        """
        Past actions taken in a given state, indexed by an integer id
        The value is a 2-tuple of state and action
        """

    def act(self, state):
        """
        :rtype: (int, str)
        :returns: An integer id for the episode, and the action
        """
        aVals = self._q[self._stateIndex(state)]

        if np.random.rand() < self._epsilon:
            actIndex = np.random.randint(len(aVals))
        else:
            actIndex = aVals.argmax()
        action = actionNames[actIndex]
        self._id += 1
        self._actRecord[self._id] = (state, action)

        return (self._id, action)

    def reward(self, id, r):
        """
        Reward the model for the action it took in a given state

        :param float           r: The reward/penalty
        :param int            id: The episode id
        """

        alpha = self._alpha
        q = self._q
        (state, action) = self._actRecord[id]
        index = self._qIndex(state, action)
        q[index] = q[index]*(1-alpha) + alpha*r;



    def _qIndex(self, state, action):
        """
        :param str state: The state of the environment
        :param str action: The action to take
        """
        stateIndex = self._stateIndex(state)
        actionIndex = actionNames.index(action)
        return (stateIndex, actionIndex)

    def _stateIndex(self, state):
        stateIndex = 0
        combsRemaining = self._numStates
        for (i, stateComp) in enumerate(stateComps):
                combsRemaining = combsRemaining // len(stateVals[i])
                stateIndex += stateVals[i].index(state[stateComp]) * combsRemaining
        return stateIndex
