# Pacman-AI-Projects

This repository hosts a comprehensive set of AI projects built on the **Pacman** game environment, adapted from UC Berkeley’s CS188: Introduction to Artificial Intelligence. These projects are designed to introduce key concepts in artificial intelligence, including search algorithms, Markov decision processes, reinforcement learning, and probabilistic inference.

---

## 📁 Repository Contents

├── autograder.py # Unified testing interface
├── bayesNet.py # Bayesian network infrastructure
├── bayesHMMTestClasses.py # HMM testing configurations
├── busters.py # Main environment for ghost-tracking agents
├── bustersAgents.py # Ghost tracking and inference agents
├── bustersGhostAgents.py # Custom ghost behavior logic
├── distanceCalculator.py # Maze distance computation utility
├── factorOperations.py # Factor joining, marginalizing, normalizing
├── game.py # Game engine (states, actions, agent control)
├── ghostAgents.py # Ghost control strategies
├── grading.py # Scoring support for autograder
├── graphicsDisplay.py # GUI visualizations
├── graphicsUtils.py # Drawing functions for GUI
├── hunters.py # Entry point for inference-based agents
├── inference.py # Core inference algorithms
├── keyboardAgents.py # Manual control agent (for testing)
├── layout.py # Maze loader and validation
├── pacman.py # Standard launcher for Pacman agents
├── pacmanAgents.py # Agent implementations for different tasks
├── projectParams.py # Default parameters and flags
├── qlearningAgents.py # Q-learning and Approximate Q-learning
├── search.py # Search algorithms (DFS, BFS, UCS, A*)
├── searchAgents.py # Agents solving navigation/search tasks
├── testClasses.py # Test scaffolding
├── testParser.py # XML test suite parser
├── textDisplay.py # Console display renderer
├── util.py # Priority queues, counters, utilities
├── valueIterationAgents.py # MDP value iteration agent logic
├── layouts/ # Maze maps for different tasks
│ ├── bigHunt.lay
│ ├── oneHunt.lay
│ ├── openHunt.lay
│ └── smallHunt.lay
├── supplemental/ # Written report files per project
│ ├── P1_answers_supplemental.txt
│ ├── P3_supplement.txt
│ └── P4_Supplement.txt
└── VERSION # Project version string


---

## Project Breakdown

### Project 1: Search

Implements uninformed and informed search algorithms:
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Uniform Cost Search (UCS)
- A* Search with heuristics

Agents solve maze navigation problems like:
- `PositionSearchProblem`
- `CornersProblem`
- `FoodSearchProblem`

Supplement: `P1_answers_supplemental.txt`

---

### Project 2: Markov Decision Processes (MDPs)

Builds agents that solve MDPs using:
- Value Iteration
- Bellman updates
- Policy extraction

Agent: `ValueIterationAgent` in `valueIterationAgents.py`

---

### Project 3: Reinforcement Learning

Teaches agents through interaction with the environment using:
- Q-learning
- Epsilon-greedy exploration
- Approximate Q-learning with feature extractors

Includes:
- `QLearningAgent`
- `PacmanQAgent`
- `ApproximateQAgent`

Supplement: `P3_supplement.txt`

---

### Project 4: Probabilistic Inference (Ghostbusters)

Uses Bayesian networks and Hidden Markov Models (HMMs) to track ghost locations in noisy environments. Implements:
- Exact Inference
- Particle Filtering
- Variable Elimination

Modules:
- `inference.py`
- `bayesNet.py`
- `factorOperations.py`

📝 Supplement: `P4_Supplement.txt`

---

## Getting Started

### Requirements
- Python 3.10 or 3.11
- No external packages required (uses standard Python)

### Setup
Clone this repository:
```bash
git clone https://github.com/your-username/pacman-ai-projects.git
cd pacman-ai-projects

Run Pacman with DFS SearchAgent
python pacman.py -l mediumMaze -p SearchAgent -a fn=depthFirstSearch

Run Pacman with A and Manhattan heuristic*
python pacman.py -l mediumMaze -p SearchAgent -a fn=aStarSearch,heuristic=manhattanHeuristic

Run Q-learning Agent
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

Run Ghost Tracking with Inference
python hunters.py -l bigHunt

Testing & Autograder
The autograder.py tool checks agent correctness and grading:

python autograder.py -q q1         # Test a specific question
python autograder.py -t            # Run all test cases
python autograder.py -p Project4   # Run tests for a specific project


