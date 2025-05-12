# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)


"""
In this file, you should test your Q-learning implementation on a robot crawler environment. 
It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

### Please finish the code below ##############################################
###############################################################################
    eps_min = 0.1
    eps_decay = 0.999

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES

    steps = 0

    #shorthand python instead of the for loop
    Q = [[0.0] * NUM_ACTIONS for _ in range(NUM_STATES)] 
    # same code as the gridworld.py's q-learning, im gonna put it without
    # the comments this time
    while steps < max_iterations:
        s = env.reset()
        terminal = False

        while not terminal and steps < max_iterations:
            if random.random() < eps:
                a = random.randint(0, NUM_ACTIONS - 1)
            else:
                a = max(range(NUM_ACTIONS), key=lambda x: Q[s][x])

            s_, r, terminal, _ = env.step(a)

            if terminal:
                target = r
            else:
                target = r + gamma * max(Q[s_])

            Q[s][a] = (1 - alpha) * Q[s][a] + alpha * target

            s = s_
            steps += 1
            eps = max(eps_min, eps * eps_decay)

    for s in range(NUM_STATES):
        pi[s] = max(range(NUM_ACTIONS), key=lambda a: Q[s][a])
        v[s] = max(Q[s])

    logger.log(1, v, pi)
    return pi
###############################################################################


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()