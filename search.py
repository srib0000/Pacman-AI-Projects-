# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Initializing a set to keep track of visited states
    visited = set()
    
    # Initializing a stack for depth-first search
    stack = util.Stack()
    
    # Get the start state and push it onto the stack with an empty list of actions
    start_state = problem.getStartState()
    stack.push((start_state, []))  # Stating and corresponding actions

    # Performing depth-first search
    while not stack.isEmpty():
        # Pop the current state and its corresponding actions
        current_state, actions = stack.pop()

        # Checking if the current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Checking if the current state has not been visited
        if current_state not in visited:
            # Marking the current state as visited
            visited.add(current_state)
            
            # Getting successors of the current state
            successors = problem.getSuccessors(current_state)
            
            # Iterating through successors and push them onto the stack
            for successor, action, _ in successors:
                new_actions = actions + [action]
                stack.push((successor, new_actions))

    # Returning an empty list if no solution is found
    return []

   #util.raiseNotDefined()



def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # Initializing a set to keep track of visited states
    visited = set()
    
    # Initializing a queue for breadth-first search
    queue = util.Queue()
    
    # Get the start state and push it onto the queue with an empty list of actions
    start_state = problem.getStartState()
    queue.push((start_state, []))  # State and corresponding actions

    # Performing breadth-first search
    while not queue.isEmpty():
        # Dequeue the current state and its corresponding actions
        current_state, actions = queue.pop()

        # Checking if the current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Checking if the current state has not been visited
        if current_state not in visited:
            # Mark the current state as visited
            visited.add(current_state)
            
            # Getting successors of the current state
            successors = problem.getSuccessors(current_state)
            
            # Iterating through successors and enqueue them
            for successor, action, _ in successors:
                new_actions = actions + [action]
                queue.push((successor, new_actions))

    # Return an empty list if no solution is found
    return []


    #util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # Initializing a priority queue for uniform-cost search
    priorityQueue = util.PriorityQueue()
    
    # Push the start state onto the priority queue with an empty list of actions and cost 0
    priorityQueue.push((problem.getStartState(), [], 0), 0)

    # Initializing a set to keep track of visited states
    visited = set()

    # Performing uniform-cost search
    while not priorityQueue.isEmpty():
        # Pop the current state, its corresponding actions, and the cost
        state, actions, cost = priorityQueue.pop()

        # Checking if the current state is the goal state
        if problem.isGoalState(state):
            return actions

        # Checking if the current state has not been visited
        if state not in visited:
            # Mark the current state as visited
            visited.add(state)
            
            # Getting successors of the current state
            successors = problem.getSuccessors(state)
            
            # Iterating through successors and push them onto the priority queue
            for successor, action, stepCost in successors:
                new_cost = cost + stepCost
                priorityQueue.push((successor, actions + [action], new_cost), new_cost)

    # Return an empty list if no solution is found
    return []

    #util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Initializing a priority queue for A* search
    priority_queue = util.PriorityQueue()
    
    # Getting the start state and calculate the heuristic value
    start_state = problem.getStartState()
    start_heuristic = heuristic(start_state, problem)
    
    # Push the start state onto the priority queue with an empty list of actions and cost 0
    priority_queue.push((start_state, [], 0), start_heuristic)  # State, actions, and cost

    # Initializing a set to keep track of visited states
    visited = set()

    # Performing A* search
    while not priority_queue.isEmpty():
        # Pop the current state, its corresponding actions, and the cost
        current_state, actions, cost = priority_queue.pop()

        # Checking if the current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Checking if the current state has not been visited
        if current_state not in visited:
            # Marking the current state as visited
            visited.add(current_state)
            
            # Getting successors of the current state
            successors = problem.getSuccessors(current_state)
            
            # Iterating through successors and push them onto the priority queue
            for successor, action, step_cost in successors:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                
                # Calculating the heuristic value for the successor
                heuristic_value = heuristic(successor, problem)
                
                # Calculating the total priority (cost + heuristic) for the successor
                total_priority = new_cost + heuristic_value
                
                # Push the successor onto the priority queue
                priority_queue.push((successor, new_actions, new_cost), total_priority)

    # Return an empty list if no solution is found
    return []

    #util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
