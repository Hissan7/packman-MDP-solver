# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The skeleton of the agent here was written by Simon Parsons, based on the
# code in pacmanAgents.py. The MDP model and solver below are our own work.

# Approach
# --------
# Every move Pacman builds an MDP of the current world and solves it with
# Value Iteration:
#   States  S : every non-wall grid cell.
#   Actions A : North, South, East, West.
#   P(s'|s,a) : the api motion model (0.8 intended, 0.1 each side, walls mean stay).
#   R(s)      : rebuilt each move from food, capsules and ghosts (see buildRewards).
#   Discount  : GAMMA.
# The action played is the argmax of expected utility at Pacman's cell.

from collections import deque

from pacman import Directions
from game import Agent
import api

# Motion model (matches api.directionProb).
MOVE_PROB = 0.8

# Discount factor; high so distant food still pulls Pacman (needed for mediumMaze).
GAMMA = 0.95

# Value Iteration stopping rules.
EPSILON = 0.001
MAX_ITERATIONS = 300

# Rewards.
STEP_REWARD = -0.04          # Cost of every move, encourages short paths.
FOOD_REWARD = 10.0           # Value of a food cell.
LAST_FOOD_REWARD = 30.0      # Value of the final food, so the game is finished.
CAPSULE_REWARD = 20.0        # Value of a capsule cell.
GHOST_PENALTY = -500.0       # Penalty on a cell holding an active ghost.
SCARED_GHOST_REWARD = 50.0   # Value of a cell holding a scared (edible) ghost.

# Danger halo around active ghosts.
DANGER_RADIUS = 3            # Cells (maze distance) the halo reaches.
HALO_DECAY = 0.5             # Penalty multiplier per cell of distance.

# Unit step for each action, and the two sideways slips for each action.
STEP = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}
SLIPS = {Directions.NORTH: (Directions.WEST, Directions.EAST),
         Directions.SOUTH: (Directions.EAST, Directions.WEST),
         Directions.EAST: (Directions.NORTH, Directions.SOUTH),
         Directions.WEST: (Directions.SOUTH, Directions.NORTH)}
ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]


class MDPAgent(Agent):
    """Pacman agent that picks moves by solving an MDP with Value Iteration."""

    def __init__(self):
        """Set up empty per-game state."""
        self.resetGameState()

    def resetGameState(self):
        """Forget everything learned about the current game."""
        self.cells = []          # Non-wall cells, indexed by position in this list.
        self.index = {}          # Cell (x, y) -> index in self.cells.
        self.neighbours = []     # Per cell: list of adjacent cell indices.
        self.transitions = []    # Per cell, per action: list of (prob, next index).
        self.utilities = []      # Utility of each cell from the latest solve.

    def registerInitialState(self, state):
        """Build the parts of the MDP that never change (states, transitions)."""
        self.resetGameState()
        self.buildStates(state)
        self.buildTransitions()
        self.utilities = [0.0] * len(self.cells)

    def final(self, state):
        """Clear per-game state so the next game starts fresh."""
        self.resetGameState()

    def getAction(self, state):
        """Solve the MDP for the current world and return the best legal move."""
        legal = [a for a in api.legalActions(state) if a != Directions.STOP]
        if not legal:
            return api.makeMove(Directions.STOP, legal)
        here = self.index.get(api.whereAmI(state))
        if here is None:
            return api.makeMove(legal[0], legal)
        rewards, absorbing = self.buildRewards(state)
        self.solve(rewards, absorbing)
        return api.makeMove(self.bestAction(here, legal), legal)

    # ---- Model construction (once per game) ----

    def buildStates(self, state):
        """Collect every non-wall cell inside the grid."""
        width = api.corners(state)[3][0] + 1
        height = api.corners(state)[3][1] + 1
        walls = set(api.walls(state))
        self.cells = [(x, y) for x in range(width) for y in range(height)
                      if (x, y) not in walls]
        self.index = {cell: i for i, cell in enumerate(self.cells)}

    def move(self, cell, action):
        """Return the index reached from cell by action, or cell itself if blocked."""
        dx, dy = STEP[action]
        target = (cell[0] + dx, cell[1] + dy)
        return self.index.get(target, self.index[cell])

    def buildTransitions(self):
        """Precompute P(s'|s,a) as (probability, next index) lists."""
        slip = (1.0 - MOVE_PROB) / 2.0
        self.transitions = []
        self.neighbours = []
        for cell in self.cells:
            options = []
            for action in ACTIONS:
                left, right = SLIPS[action]
                options.append([(MOVE_PROB, self.move(cell, action)),
                                (slip, self.move(cell, left)),
                                (slip, self.move(cell, right))])
            self.transitions.append(options)
            self.neighbours.append([self.move(cell, a) for a in ACTIONS])

    # ---- Reward construction (every move) ----

    def ghostCells(self, position):
        """Return indices of the cells a (possibly fractional) ghost position touches."""
        xs = {int(position[0]), int(position[0] + 0.5)}
        ys = {int(position[1]), int(position[1] + 0.5)}
        return [self.index[(x, y)] for x in xs for y in ys if (x, y) in self.index]

    def dangerDistances(self, sources):
        """Breadth-first maze distance from the nearest source, up to DANGER_RADIUS."""
        distance = {i: 0 for i in sources}
        queue = deque(sources)
        while queue:
            i = queue.popleft()
            if distance[i] == DANGER_RADIUS:
                continue
            for j in self.neighbours[i]:
                if j not in distance:
                    distance[j] = distance[i] + 1
                    queue.append(j)
        return distance

    def buildRewards(self, state):
        """Build R(s) from the current world; also flag cells that end the episode."""
        rewards = [STEP_REWARD] * len(self.cells)
        absorbing = [False] * len(self.cells)

        food = [self.index[f] for f in api.food(state) if f in self.index]
        foodReward = LAST_FOOD_REWARD if len(food) == 1 else FOOD_REWARD
        for i in food:
            rewards[i] += foodReward
        for c in api.capsules(state):
            if c in self.index:
                rewards[self.index[c]] += CAPSULE_REWARD

        active, scared = [], []
        for position, timer in api.ghostStatesWithTimes(state):
            (scared if timer > 0 else active).extend(self.ghostCells(position))
        for i in scared:
            rewards[i] += SCARED_GHOST_REWARD
        for i, d in self.dangerDistances(active).items():
            rewards[i] += GHOST_PENALTY * HALO_DECAY ** d
        for i in active:
            rewards[i] = GHOST_PENALTY
            absorbing[i] = True
        return rewards, absorbing

    # ---- Solver ----

    def solve(self, rewards, absorbing):
        """Run in-place Value Iteration, warm-started from the previous utilities."""
        utilities = self.utilities
        for i, flag in enumerate(absorbing):
            if flag:
                utilities[i] = rewards[i]
        for _ in range(MAX_ITERATIONS):
            change = 0.0
            for i, options in enumerate(self.transitions):
                if absorbing[i]:
                    continue
                best = max(sum(p * utilities[j] for p, j in option)
                           for option in options)
                updated = rewards[i] + GAMMA * best
                change = max(change, abs(updated - utilities[i]))
                utilities[i] = updated
            if change < EPSILON:
                break

    def bestAction(self, here, legal):
        """Return the legal action with the highest expected utility from here."""
        best, bestValue = legal[0], None
        for action in legal:
            value = sum(p * self.utilities[j]
                        for p, j in self.transitions[here][ACTIONS.index(action)])
            if bestValue is None or value > bestValue:
                best, bestValue = action, value
        return best
