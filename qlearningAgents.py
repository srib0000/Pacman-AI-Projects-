# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # set all (state, action) pair to 0
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.QValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # Getting legal actions for the current state
        valid_actions = self.getLegalActions(state)
        # If no legal actions available (terminal state), then return 0.0
        if len(valid_actions) == 0:
         return 0.0
        # Getting the action with the highest Q-value according to the current policy
        finest_action = self.getPolicy(state)
        # Returning the Q-value of the state-action pair with the finest action
        return self.getQValue(state, finest_action)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Getting legal actions for the current state
        valid_actions= self.getLegalActions(state)
        # If no legal actions available (terminal state), then return None
        if len(valid_actions) == 0:
            return None
        action_values = {}
        finest_Q_value = float('-inf')
        # Iterating through each legal action and calculate its Q-value
        for action in valid_actions:
            goal_Q_value = self.getQValue(state, action)
            action_values[action] = goal_Q_value
            # Tracking the maximum Q-value and corresponding actions
            if goal_Q_value > finest_Q_value:
                finest_Q_value = goal_Q_value
        # Getting actions with the highest Q-value
        finest_actions = [y for y,z in action_values.items() if z == finest_Q_value]
        # Returning a random choice among the finest actions
        return random.choice(finest_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Getting valid actions for the current state
        valid_actions = self.getLegalActions(state)
        # If no vallid actions available (terminal state), then return None
        if len(valid_actions) == 0:
            return None
        action = None
        # With probability epsilon, choosing a random action; otherwise, choosing the best policy action
        if not util.flipCoin(self.epsilon):
            action = self.getPolicy(state)
        else:
            action = random.choice(valid_actions)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Getting the previous Q-value for the state-action pair
        previous_value = self.getQValue(state,action)
        # Getting the value of the next state
        succeeding_value = self.getValue(nextState)
        # Updating the Q-value using the Q-learning update rule
        next_value = previous_value + self.alpha * \
        (reward + (self.discount * succeeding_value) - previous_value)
        # Updating the Q-value table with the new value
        self.QValues[(state, action)] = next_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Extracting features for the given state-action pair
        features = self.featExtractor.getFeatures(state, action)
        Q_value = 0
        # Computing Q-value as the dot product of feature weights and feature values
        for feature in features:
            Q_value += self.weights[feature] * features[feature]
        return Q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # Extracting features for the given state-action pair
        features = self.featExtractor.getFeatures(state, action)
        # Computing the difference between the estimated Q-value and the observed reward
        variance = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        # Updating weights using the gradient descent algorithm
        for feature in features:
            self.weights[feature] += self.alpha * variance * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
        # you might want to print your weights here for debugging
           print("Training completed. Final weights:")
           print(self.weights)
           pass