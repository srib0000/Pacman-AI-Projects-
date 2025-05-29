# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information that can be extracted from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
        # Calculating distance to the nearest food
        nearestFoodDistance = self.calculateNearestFoodDistance(newPos, newFood)

        # Calculating distance to the nearest scared ghost
        nearestScaredGhostDistance = self.calculateNearestScaredGhostDistance(newPos, newGhostStates)

        # Using a weighted sum of the distances and the current score
        score = successorGameState.getScore() - 0.5 * nearestFoodDistance + 2 * nearestScaredGhostDistance

        return score

    def calculateNearestFoodDistance(self, position, foodGrid):
        
        # Calculating distances from the current position to all remaining food positions
        foodDistances = [manhattanDistance(position, foodPos) for foodPos in foodGrid.asList()]
        return min(foodDistances) if foodDistances else 0

    def calculateNearestScaredGhostDistance(self, position, ghostStates):
        
        # Calculating the distance to the nearest scared ghost if there are any within a certain range
        minGhostDistance = float('inf')
        for ghostState in ghostStates:
            # Skipping scared ghosts as they are not a threat
            if ghostState.scaredTimer > 0:
                continue
            # Calculating Manhattan distance to the ghost
            ghostDistance = manhattanDistance(position, ghostState.getPosition())
            # Updating the minimum distance
            minGhostDistance = min(minGhostDistance, ghostDistance)
        # Returning the minimum distance or zero if there are no scared ghosts are within the specified range
        return minGhostDistance if minGhostDistance < 5 else 0


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def min_agent(state, level, agent):
    # The base case is if the current agent exceeds the total number of agents,
    # check if the depth limit is reached. If reached, return the evaluation function,
    # otherwise, continue the search with the next level.
            if agent == state.getNumAgents():
                if level == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return min_agent(state, level + 1, 0)
            else:
        # For each legal action for the current agent,
        # recursively calculate the value of the successor state.
                actions = state.getLegalActions(agent)
        # If no legal actions are available, return the evaluation function for the current state.
                if len(actions) == 0:
                    return self.evaluationFunction(state)
        # Generate the successor states for each legal action and
        # recursively calculate the min value for the next level.
                next_states = (
                    min_agent(state.generateSuccessor(agent, action),
                    level, agent + 1)
                    for action in actions
                    )
        # Return the maximum or minimum value based on whether the current agent is Pacman or a ghost.
                return (max if agent == 0 else min)(next_states)
            
# The main function to find the best action for Pacman using minimax algorithm.
# It evaluates all legal actions and selects the one with the maximum min value.

        return max(
            gameState.getLegalActions(0),
            key = lambda x: min_agent(gameState.generateSuccessor(0, x), 1, 1)
            )
    
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # If reached the specified depth or the state is a win/lose state,
        # return the evaluation function value for the state.
        def maximum_value(state, depth, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        # Initializing alpha to negative infinity.
            a = float('-inf')
        # Iterate over legal actions for the first agent (0).
            for action in state.getLegalActions(0):
        # Generating successor state after taking the action.
                successor = state.generateSuccessor(0, action)
        # Updating alpha with the maximum value between the current alpha and
        # the result of the minimum_value function for the successor state.
                a= max(a, minimum_value(successor, 1, depth, alpha, beta))
        # If alpha is greater than beta, prune the search and return alpha.
                if a > beta:
                    return a
        # Updating alpha for pruning in the next iteration.
                alpha = max(alpha, a)
            return a

        def minimum_value(state, agent, depth, alpha, beta):
            # If reached the specified depth or the state is a win/lose state,
            # return the evaluation function value for the state.
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
    
            # Initializing beta to positive infinity.
            a = float('inf')
            # Iterating over legal actions for the current agent.
            for action in state.getLegalActions(agent):
            # Generating successor state after taking the action.
                successor = state.generateSuccessor(agent, action)
                # If the current agent is the last agent, update a with the minimum value
                # between the current a and the result of the maximum_value function for
                # the successor state at the next depth.
                if agent == state.getNumAgents() - 1:
                    a = min(a, maximum_value(successor, depth + 1, alpha, beta))
                # If the current agent is not the last agent, update a with the minimum
                # value between the current a and the result of the minimum_value function
                # for the successor state with the next agent.
                else:
                    a = min(a, minimum_value(successor, agent + 1, depth, alpha, beta))
                # If a is less than alpha, prune the search and return a.
                if a < alpha:
                    return a
                
                beta = min(beta, a)
            return a

        # Getting legal actions for the current player
        actions = gameState.getLegalActions(0)
        # Initializing variables
        best_action = None # Variable to store the best action
        alpha = float('-inf')  # Alpha value for alpha-beta pruning (initialized to negative infinity)
        beta = float('inf')  # Beta value for alpha-beta pruning (initialized to positive infinity)
        # Loop through each legal action
        for action in actions:
            # Generating successor game state after taking the current action
            successor = gameState.generateSuccessor(0, action)
            # Calling the function to calculate the minimum value
            # for the opponent player, using alpha-beta pruning
            value = minimum_value(successor, 1, 0, alpha, beta)
            # Updating alpha if the calculated value is greater than the current alpha
            if value > alpha:
                alpha = value
                best_action = action # Update the best action
        # Return the best action found
        return best_action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Getting the total number of agents
        pac = gameState.getNumAgents()

        def expectimaxMaximize(state: GameState, agentIndex, level):
            """
            Maximize function for the expectimax agent.

            Args:
            - state: Current game state
            - agentIndex: Index of the current agent
            - level: Depth level in the search tree

            Returns:
            - The best action for the current agent based on the expectimax algorithm.
            """
            properAction = None
            actions = state.getLegalActions(agentIndex)
            maxVal = float('-inf')

            for action in actions:
            # Generating successor state for each action
                successor = state.generateSuccessor(agentIndex, action)
            # Calculating the value using the average function
                value = expectimaxAverageFunction(successor, agentIndex + 1, level)
                
                if value > maxVal:
                    maxVal = value
                    properAction = action

            return properAction

        def expectimaxAverageFunction(state: GameState, agentIndex, depth):
            """
            Average function to calculate the expected value for the expectimax agent.

            Args:
            - state: Current game state
            - agentIndex: Index of the current agent
            - depth: Depth level in the search tree

            Returns:
            - The expected value for the current agent based on the expectimax algorithm.
            """
            # Checking if reached the specified depth or terminal state
            if (depth == self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)

            if agentIndex == 0:
                # Pacman's turn where maximization happens
                expected = float('-inf')

                for action in actions:
                    # Generating successor state for each action
                    successor = state.generateSuccessor(agentIndex, action)
                    # Recursiving call to the average function for the next agent
                    expected = max(expected, expectimaxAverageFunction(successor, 1, depth))

                return expected
            else:
                # Ghost's turn and it is the average
                expected = 0

                for action in actions:
                    # Generating successor state for each action
                    successor = state.generateSuccessor(agentIndex, action)

                    if agentIndex < pac - 1:
                    # Recursively calling to the average function for the next ghost agent
                        expected += expectimaxAverageFunction(successor, agentIndex + 1, depth)
                    else:
                    # Recursively calling to the average function for Pacman in the next depth
                        expected += expectimaxAverageFunction(successor, 0, depth + 1)
                        
                # Calculating the average of all possible outcomes
                expected /= len(actions)
                return expected

        # Starting the expectimax algorithm from Pacman's perspective
        return expectimaxMaximize(gameState, self.index, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Extracting relevant information from the current game state
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    currentScore = currentGameState.getScore()

    # Here Checking if the game is in a losing or winning state takes place
    if currentGameState.isLose():
        return float("-inf")
    if currentGameState.isWin():
        return float("inf")

    # Calculating the minimum distance to the closest food
    minimumdistancetofood = float("inf")
    newfoodlocation = newFood.asList()
    for foodPos in newfoodlocation:
        distancetofood = manhattanDistance(newPos, foodPos)
        if distancetofood < minimumdistancetofood:
            minimumdistancetofood = distancetofood
            
   # Counting the number of remaining food pellets and capsules
    numoffoodleft = len(newfoodlocation)

    numoftabsleft = len(currentGameState.getCapsules())
    
    # Separating scared and non-scared ghosts
    scaredghosts, terrorghosts = [], []
    for ghoststate in newGhostStates:
        if ghoststate.scaredTimer:
            scaredghosts.append(ghoststate)
        else:
            terrorghosts.append(ghoststate)
    # Calculating the minimum distance to the closest scared ghost if there is any
    minimumdistancetoscaredghost = 0
    # Calculating the minimum distance to the closest non-scared ghost if there is any
    minimumdistancetoterrorghost = float("inf")
    if scaredghosts:
        for scaredGhost in scaredghosts:
            distancetoscaredghost = manhattanDistance(newPos, scaredGhost.getPosition())
            if distancetoscaredghost < minimumdistancetoscaredghost:
                minimumdistancetoscaredghost = distancetoscaredghost

    if terrorghosts:
        for terrorghost in terrorghosts:
            distanceToTerroGhost = manhattanDistance(newPos, terrorghost.getPosition())
            if distanceToTerroGhost < minimumdistancetoterrorghost :
                minimumdistancetoterrorghost  = distanceToTerroGhost

    # Calculating the final score based on all the factors
    score = currentScore + 15.0 / minimumdistancetofood - 25.0 / minimumdistancetoterrorghost - 100 * minimumdistancetoscaredghost - 70 * numoftabsleft - 15 * numoffoodleft
    # Returning the calculated score
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

