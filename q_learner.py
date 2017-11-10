"""
Author:     Arnold YS Yeung
Date:       October 28th, 2017
This contains a Q-Learner class for training and querying a Q-table.

2017-11-11: Completed and tested qLearner class in a maze-solving application
            (see test_q_learner.py) - dyna not implemented
            (Yes. I have been lazy/busy recently.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import random as rdm

class qLearner():

    def __init__(self,  \
                 num_states,        #   number of states to consider        \            
                 num_actions,       #   number of actions available         \            
                 alpha = 0.2,       #   learning rate (0 to 1.0)            \            
                 gamma = 0.9,       #   discount rate (0 to 1.0)            \            
                 rar = 0.5,         #   random action rate (0 to 1.0)       \           
                 radr = 0.99,       #   random action decay rate (0 to 1.0) \            
                 dyna = 1,          \
                 verbose = True):
        #   define input properties
        self.num_states = num_states;
        self.num_actions = num_actions;
        self.alpha = alpha;
        self.gamma = gamma;
        self.rar = rar;
        self.radr = radr;
        self.dyna = dyna;
        self.verbose = verbose;

        #   initial state and action
        self.s = 0;
        self.a = 0;

        #   Q[s,a] = immediate reward + discounted reward for s,a
        self.Q = np.zeros((self.num_states, self.num_actions));
        #   This Q-table contains the Q-value for state x action

    def best_action(self, s):
    #   Returns best action (policy) to maximize sum of immediate and discounted
    #   reward
        return np.argmax(self.Q[s, :]);     
                
    def query(self, s_prime, r):
    #   Keeps previous state (s) and action (a) and updates q_table based on
    #   the inputted new state (s_prime) and reward (r) with a new experience
    #   tuple <s, a, s_prime, r>.  Returns the next action (best or random).
    #   Note that in reality, s_prime and r are observed from state s after
    #   action a is taken.  This function is a training function.
    #   For example, s = 1, a = D, s_prime = 4, reward = 102.
    #   We take action D while in state 1.  We observe the outcome to be
    #   state 4 with a reward of 102. The calculated next action to take
    #   is returned and based solely on state 4.

        prev_state = self.s;
        prev_action = self.a;
        new_state = s_prime;
        reward = r;

        if rdm.uniform(0, 1) < self.rar:             #   if random action is selected
            next_action = rdm.randint(0, self.num_actions-1);       #   select random action
            #print('Random action selected...');
        else:
            next_action = self.best_action(new_state);
            #print('Best action selected using policy...');

        #   update Q-table with Q[s,a] where s = prev_state, a = prev_action
        self.Q[prev_state, prev_action] = (1-self.alpha)*self.Q[prev_state, prev_action] \
                                          + self.alpha * (reward + self.gamma * \
                                                          self.Q[new_state, self.best_action(new_state)]);

        self.rar = self.radr * self.rar;        #   reduce random action rate

        #   update state and action
        self.a=next_action;
        self.s=s_prime;
        
        return next_action;
    
    def querySetState(self, s):
    #   Keeps previous state (s) as new state.  Returns the next action
    #   to take at state s.  Does not update Q-table or reduce random action rate.
    #   2 main uses:
    #       - Set initial state of learner
    #       - Use a learned policy without updating Q-table
        self.s = s;
        if rdm.uniform(0, 1) > self.rar:             #   if random action is selected
            next_action = rdm.randint(0, self.num_actions-1);       #   select random action
            #print('Random action selected...');
        else:
            next_action = self.best_action(s);
            #print('Best action selected using policy...');
    
        return next_action;
