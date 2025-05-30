# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached, 
the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the infinity norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
#Ashton
    for iteration in range(1, max_iterations + 1):
        v_new = v[:]
        #Bellman update
        for s in range(NUM_STATES):
            max_value = float('-inf')
            best_action = None
            
            #Look through all actions to find the best value
            for a in range(NUM_ACTIONS):
                value = 0
                for p, s_, r, terminal in TRANSITION_MODEL[s][a]:
                    value += p * (r + gamma * v[s_] * (1 - terminal))
                
                if value > max_value:
                    max_value = value
                    best_action = a
            
            #Update the value
            v_new[s] = max_value
            pi[s] = best_action 
        
        #If the value function has converged
        if max(abs(v_new[i] - v[i]) for i in range(NUM_STATES)) < 1e-4:
            print(f"Value iteration converged after {iteration} iterations.")
            break
        
        v = v_new[:]
        
        logger.log(iteration, v, pi)
###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the infinity norm 
        norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of values by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
#Ashton
    for iteration in range(1, max_iterations + 1):
        #Calculate the value function
        while True:
            v_new = v[:]
            #Bellman update
            for s in range(NUM_STATES):
                action = pi[s]
                value = 0
                for p, s_, r, terminal in TRANSITION_MODEL[s][action]:
                    value += p * (r + gamma * v[s_] * (1 - terminal))
                v_new[s] = value

            #Check for convergence
            if max(abs(v_new[i] - v[i]) for i in range(NUM_STATES)) < 1e-4:
                break
            v = v_new[:]
        
        #Update the policy
        policy_stable = True
        for s in range(NUM_STATES):
            old_action = pi[s]
            max_value = float('-inf')
            best_action = None
            for a in range(NUM_ACTIONS):
                value = 0
                for p, s_, r, terminal in TRANSITION_MODEL[s][a]:
                    value += p * (r + gamma * v[s_] * (1 - terminal))
                if value > max_value:
                    max_value = value
                    best_action = a
            pi[s] = best_action

            #If the policy is not stable, continue the process. Else stop
            if old_action != pi[s]:
                policy_stable = False

        if policy_stable:
            print(f"Policy iteration converged after {iteration} iterations.")
            break

        logger.log(iteration, v, pi)
###############################################################################
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        ** note from nate --- this is a boundary that can be applied to episodes & steps
           its a "GLOBAL TRAINING LIMIT"
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
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    steps = 0

### Please finish the code below ##############################################
###############################################################################
    # Nate --- WARNING IMMA COMMENT A LOT TO UNDERSTAND

    eps_min = 0.1
    eps_decay = 0.999 # exponential decay

    # Initialize Q(s,a) for all s, a
    Q = []
    for i in range(NUM_STATES):
        row = [0.0] * NUM_ACTIONS
        Q.append(row)
    #repeat for each episode
    while steps < max_iterations:
        #get initial state s
        s = env.reset() # env API call to reset environment
        terminal = False # we in start state so its never terminal
        #repeat for each step
        while not terminal and steps < max_iterations:
            #sample action a from s, observe reward r and next state s_
            if random.random() < eps: # epsilon greedy action selection || probability epsilon
                # pick a random action from the range of NUM_ACTIONS
                a = random.randint(0, NUM_ACTIONS - 1) 
            else: # probability 1-epsilon
                # among all valid actions [0,1,...,n-1], choose the one x that gives 
                # the highest Q[s][x]
                a = max(range(NUM_ACTIONS), key=lambda x: Q[s][x])
                # explanation of line of code above ^^
                # Why range?? --- creates a list to choose from
                # of all possible action indices
                # Lambda --- for each action x (like 0,1,2...), evaluate 
                # Q[s][x] and use that value to determine which x is the best

            #take action
            s_, r, terminal, _ = env.step(a)

            #compute target for Q-update
            if terminal:
                target = r # no future reward
            else:
                #Bellman target
                target = r + gamma * max(Q[s_])

            #update Q-value
            Q[s][a] = (1-alpha) * Q[s][a] + alpha * target # Q-learning update formula

            s = s_
            steps += 1

            #epsilon decay
            eps = max(eps_min, eps * eps_decay)
            #for GUI
            if steps % 500 == 0 or steps == 1:
                for state in range(NUM_STATES):
                    v[state] = max(Q[state])
                    pi[state] = max(range(NUM_ACTIONS), key=lambda a: Q[state][a])
                logger.log(steps, v, pi)

    #extract policy and value function
    for s in range(NUM_STATES):
        pi[s] = max(range(NUM_ACTIONS), key=lambda a: Q[s][a])
        v[s] = max(Q[s])

    logger.log(steps,v,pi)
###############################################################################
    return pi



if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q-Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()