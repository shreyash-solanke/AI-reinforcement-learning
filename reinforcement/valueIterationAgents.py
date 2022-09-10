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
import math

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        for i in range(self.iterations):
            new_values = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    if len(actions) > 0:  # not terminal state
                        # max_action_val = max([self.calculateActionVal(state, action) for action in actions])
                        # new_values[state] = self.mdp.getReward() + self.discount * max_action_val
                        new_values[state] = max([self.computeQValueFromValues(state, action) for action in actions])
                    else:
                        new_values[state] = self.mdp.getReward(state, 'exit', '')
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        tsp_list = self.mdp.getTransitionStatesAndProbs(state, action)
        q_arr = []
        for next_state, prob in tsp_list:
            # val = prob * (self.mdp.getReward(state, action, next_state) + self.discount*self.values[next_state])
            val = self.mdp.getReward(state, action, next_state) + self.discount * (prob * self.values[next_state])
            q_arr.append(val)
        # print(q_arr)
        return sum(q_arr)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        stateAction = util.Counter()
        for a in self.mdp.getPossibleActions(state):
            stateAction[a] = self.computeQValueFromValues(state, a)
        policy = stateAction.argMax()
        return policy

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
        states = self.mdp.getStates()
        number_of_states = len(states)
        for i in range(self.iterations):
            print("i: ", i)
            state = states[i%number_of_states]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                if len(actions) > 0:  # not terminal state
                    self.values[state] = max([self.computeQValueFromValues(state, action) for action in actions])

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

        state_predeccors = {}
        # compute predecessors for each set
        for to_state in self.mdp.getStates():
            _set = set()
            for from_state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(from_state)
                for action in actions:
                    ts_list = self.mdp.getTransitionStatesAndProbs(from_state, action)
                    for next_state, prob in ts_list:
                        if next_state == to_state:
                            _set.add(from_state)
                            break
            state_predeccors[to_state] = _set

        # print(state_predeccors)

        # Initialize an empty priority queue
        state_diff_pq = util.PriorityQueue()

        # prioritizing highest difference
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                curr_val = self.values[state]
                highesh_q_val = max([self.computeQValueFromValues(state, action)
                                     for action in self.mdp.getPossibleActions(state)])
                diff = abs(highesh_q_val - curr_val)
                state_diff_pq.push(state, -diff)

        # Prioritized Sweeping Value Iteration
        for i in range(self.iterations):
            if state_diff_pq.isEmpty():
                break
            state = state_diff_pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = max(
                        [self.computeQValueFromValues(state, action)
                         for action in self.mdp.getPossibleActions(state)])
                for each_predecessor in state_predeccors[state]:
                    curr_val = self.values[each_predecessor]
                    highesh_q_val = max(
                        [self.computeQValueFromValues(each_predecessor, action)
                         for action in self.mdp.getPossibleActions(each_predecessor)])
                    diff = abs(highesh_q_val - curr_val)
                    if diff > self.theta:
                        state_diff_pq.update(each_predecessor, -diff)
