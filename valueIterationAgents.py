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

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
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
        # Loop for a given number of iterations
        for _ in range(self.iterations):
            new_values = util.Counter()  # A new set of values for the states
            for state in self.mdp.getStates():  # Loop over all states
                if not self.mdp.isTerminal(state):  # If not a terminal state
                    max_value = float("-inf")  # Initializing max_value to negative infinity
                    # Loop over all possible actions for the current state
                    for action in self.mdp.getPossibleActions(state):
                        q_value = self.computeQValueFromValues(state, action)  # Computing Q-value for the action
                        max_value = max(max_value, q_value)  # Updating max_value
                    new_values[state] = max_value  # Storing the maximum Q-value for the state
            self.values = new_values  # Updating the values for the next iteration

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
        q_value = 0.0  # Initializing Q-value to 0
        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)  # Getting reward for the transition
            # Computing Q-value using Bellman equation
            q_value += probability * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):  # If terminal state, return None
            return None

        best_action = None
        best_value = float("-inf")  # Initialize best_value to negative infinity

        for action in self.mdp.getPossibleActions(state):  # Loop over possible actions
            q_value = self.computeQValueFromValues(state, action)  # Compute Q-value for action
            if q_value > best_value:  # Update best_action if Q-value is greater
                best_value = q_value
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        states = self.mdp.getStates()  # Getting all states
        border = util.PriorityQueue()  # Priority queue to store states by variance
        previous = {}  # Dictionary to store previous states

        for m in states:
            self.values[m] = 0  # Initializing values to 0
            previous[m] = self.get_previous(m)  # Getting previous states for each state

        for m in states:
            terminal = self.mdp.isTerminal(m)  # Checking if state is terminal

            if not terminal:
                states_present_value = self.values[m]  # Current value of the state
                variance = abs(states_present_value - self.highest_value_of_Q(m))  # Computing variance
                border.push(m, -variance)  # Pushing the state into the priority queue based on variance

        for _ in range(self.iterations):  # Iterating for a given number of times

            if border.isEmpty():  # If priority queue is empty, exit
                return

            m = border.pop()  # Popping the state with the highest variance
            self.values[m] = self.highest_value_of_Q(m)  # Updating value of the state

            for n in previous[m]:  # Updating variance for previous states
                variance = abs(self.values[n] - self.highest_value_of_Q(n))
                if variance > self.theta:  # If variance is greater than theta, update priority queue
                    border.update(n, -variance)

    def highest_value_of_Q(self, state):
        return max([self.getQValue(state, x) for x in self.mdp.getPossibleActions(state)])

    def get_previous(self, state):
        previous_set = set()  # Initializing set to store previous states
        states = self.mdp.getStates()  # Getting all states
        advance = ['north', 'south', 'east', 'west']  # Possible actions

        if not self.mdp.isTerminal(state):  # If state is not terminal

            for n in states:  # Loop over all states
                terminal = self.mdp.isTerminal(n)  # Checking if state is terminal
                valid_actions = self.mdp.getPossibleActions(n)  # Get possible actions for the state

                if not terminal:

                    for go in advance:  # Loop over possible actions

                        if go in valid_actions:  # If action is valid
                            change = self.mdp.getTransitionStatesAndProbs(n, go)  # Getting transition states and probabilities

                            for m_odd, r in change:  # Loop over transition states
                                if (m_odd == state) and (r > 0):  # If transition leads to the given state and probability is positive
                                    previous_set.add(n)  # Adding the state to previous_set
        return previous_set  # Returning set of previous states

