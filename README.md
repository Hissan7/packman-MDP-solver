MDP-based Pacman agent. Each move, Pacman models the world as a Markov Decision Process (grid states, N/S/E/W actions, stochastic transitions, tuned rewards for food, capsules and ghosts) and solves it with Value Iteration to pick its action. 

Targets: win smallGrid and mediumClassic, solve mediumMaze. (Python 3, no NumPy)