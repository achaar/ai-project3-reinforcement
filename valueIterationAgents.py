# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import random

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print('runValueIteration ', self.values)
        # prevValues = self.values
        for i in range(self.iterations):
            mdpStates = self.mdp.getStates()
            newValues = {}
            for state in mdpStates:
                legalActions = self.mdp.getPossibleActions(state)
                #what if legalActions is empty
                maxValue = 0
                if legalActions:
                    values = [self.computeQValueFromValues(state, action) for action in legalActions]
                    maxValue = max(values)

                # maxIndexes = [i for i in len(values) if values[i] is maxValue]
                newValues[state] = maxValue

            for state in mdpStates:
                self.values[state] = newValues[state]

        return



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values.get(state, 0)


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # print('computeQValueFromValues ')
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += prob*(self.mdp.getReward(state, action, nextState) + self.discount*self.getValue(nextState))
            # print("computeQValueFromValues ", nextState, prob, qValue)

        return qValue
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # print('computeActionFromValues ')
        legalActions = self.mdp.getPossibleActions(state)
        if not legalActions:
            return None
        # maxValue = 0
        # maxIndex = None
        # for i in range(len(actions)):
        #     action = legalActions[i]
        #     value = self.computeQValueFromValues(state, action)
        #     if value>max_value:
        #         max_value = value
        #         max_index = i

        #implement random choice for equal values
        values = [self.computeQValueFromValues(state, action) for action in legalActions]
        maxValue = max(values)
        maxIndexes = [i for i in range(len(values)) if values[i] is maxValue]
        actionTaken = legalActions[random.choice(maxIndexes)]

        return actionTaken
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdpStates = self.mdp.getStates()
        # current_iteration = 0
        numStates = len(mdpStates)
        for i in range(self.iterations):
            state = mdpStates[i%numStates]
            legalActions = self.mdp.getPossibleActions(state)
            # what if legalActions is empty
            maxValue = 0
            if legalActions:
                values = [self.computeQValueFromValues(state, action) for action in legalActions]
                maxValue = max(values)

            # maxIndexes = [i for i in len(values) if values[i] is maxValue]
            self.values[state] = maxValue

        return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        mdpStates = self.mdp.getStates()
        pq = util.PriorityQueue()
        # print(bcolors.HEADER, 'mdpStates ', mdpStates, bcolors.ENDC)
        for state in mdpStates:
            legalActions = self.mdp.getPossibleActions(state)
            maxValue = -float('inf')  # because we are taking max over all states from this state
            # print(bcolors.OKBLUE, 'state ', state,' legalActions ', legalActions, bcolors.ENDC)
            for action in legalActions:
                # nextStates, probs = self.mdp.getTransitionStatesAndProbs(state, action)
                # print('state ', state, ' action ', action,' (nextState, prob) in ', self.mdp.getTransitionStatesAndProbs(state, action))
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if not predecessors.get(nextState):
                        predecessors[nextState] = set()
                        # print('creating set for ', nextState)
                    if prob>0:
                        predecessors[nextState].add(state)
                        # print('predecessor of ', nextState, ' is', state)

                value = self.computeQValueFromValues(state,action)
                if value>maxValue:
                    maxValue = value
            if state is not 'TERMINAL_STATE':
                diff = abs(self.getValue(state) - maxValue)
                pq.push(state, -diff)
                # print('pushing ', state, ' diff = ', self.getValue(state), ' - ',maxValue)

        for i in range(self.iterations):
            # print('iteration ', i)
            if pq.isEmpty() is True:
                return

            state = pq.pop()

            if state is not 'TERMINAL_STATE':
                legalActions = self.mdp.getPossibleActions(state)
                # what if legalActions is empty
                maxValue = 0  # default value, actual comes from self.computeQValueFromValues(state, action)
                if legalActions:
                    values = [self.computeQValueFromValues(state, action) for action in legalActions]
                    # print('values ', values)
                    maxValue = max(values)

                # maxIndexes = [i for i in len(values) if values[i] is maxValue]
                self.values[state] = maxValue
                parents = predecessors[state]
                # print(bcolors.WARNING, 'state ', state, ' maxValue ', maxValue, ' parents ', parents, bcolors.ENDC)
                for parent in parents:
                    legalActions = self.mdp.getPossibleActions(parent)
                    maxValue = 0
                    if legalActions:
                        values = [self.computeQValueFromValues(parent, action) for action in legalActions]
                        # print('child values ', values)
                        maxValue = max(values)

                    diff = abs(self.values[parent] - maxValue)
                    if diff>self.theta:
                        pq.update(parent, -diff)
                        # print(bcolors.OKGREEN, 'updating ', parent, ' with ', -diff, bcolors.ENDC)

        return





