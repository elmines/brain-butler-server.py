import numpy as np

stateComps = ["errp", "orientation", "brightness"]
stateVals = [[False, True], ["portrait", "landscape"], ["full", "medium", "low"]]
actionNames = ["nothing", "rotate", "brighten"]

class Agent(object):

    def __init__(self, initState, alpha = 0.1, epsilon = 0.2, gamma = 0.5):
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma

        self._numActions = len(actionNames)
        self._numStates = sum(len(valSet) for valSet in stateVals)

        self._q = np.zeros( (self._numStates, self._numActions) )

        self._initialized = False
        self._lastState = {}
        self._lastAction = "nothing"

    @property
    def initialized(self):
        return all(map(lambda comp: comp in self._lastState, stateComps))


    def setInitial(self, state, value):
        if state not in self._lastState:
            self._lastState[state] = value
        return self.initialized


    def reward(self, r, newS):
        """
        Reward the model and notify it of the new state newS.
        The model updates its Q-Values according to the SARSA algorithm

        :param float    r: The reward/penalty
        :param dict  newS: The new state

        :returns: The new action that the model performs in light of newS
        :rtype: str
        """
        alpha = self._alpha
        gamma = self._gamma
        q = self._q

        newA = self._chooseAction(newS)

        qIndex = self._qIndex(self._lastState, self._lastAction)
        nextQIndex = self._qIndex(newS, newA)

        #The Q-VAlue update step from SARSA
        q[qIndex] += alpha * (r + gamma*q[nextQIndex] - q[qIndex])

        self._lastState = newS
        self._lastAction = newA
        return self._lastAction

    #DO NOT alter object state with this method
    def _chooseAction(self, state):
        aVals = self._q[self._stateIndex(state)]

        if np.random.rand() < self._epsilon:
            return np.random.randint(len(aVals))
        return aVals.argmax()


    def _qIndex(self, state, action):
        stateIndex = self._stateIndex(state)
        actionIndex = actionNames.find(action)
        return (stateIndex, actionIndex)

    def _stateIndex(self, state):
        stateIndex = 0
        combsRemaining = self._numStates
        for (stateComp, i) in enumerate(stateComps):
                combsRemaining = combsRemaining / len(stateVals[i])
                stateIndex += stateVals[i].find(state[stateComp]) * combsRemaining
        return stateIndex
